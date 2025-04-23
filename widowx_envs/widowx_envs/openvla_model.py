from collections import defaultdict
from typing import Optional, Sequence
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union


import matplotlib.pyplot as plt
import numpy as np
import torch
from transforms3d.euler import euler2axangle
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import transformers
import torch
from transformers import AutoTokenizer
import pdb
from PIL import Image

text_model = "meta-llama/Llama-2-7b-chat-hf"

t_tokenizer = AutoTokenizer.from_pretrained(text_model)
class OPENVLAInference:
    def __init__(
        self,
        image_width: int = 224,
        image_height: int = 224,
        action_scale: float = 1.0,
        device_id: int = 1,
        policy_setup: str = "widowx_bridge",
        model_id_or_path: Optional[str] = None,
    ) -> None:
        
        self.model_id = model_id_or_path
        self.input_processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        device_id = 1
        self.device_id = device_id
            
        self.policy = AutoModelForVision2Seq.from_pretrained(
            self.model_id, 
            # attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,  # [Optional] Use `torch.bfloat16` for faster inference
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto"
        )
        
        self.unnorm_key = list(self.policy.norm_stats.keys())[0]
        print(self.unnorm_key)

        # Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub) ```
        self.vlm = self.policy
        self.vlm_features = []
        self.feats = []
        self.image_width = image_width
        self.image_height = image_height
        self.action_scale = action_scale

        self.observation = None
        self.tfa_time_step = None
        self.policy_state = None
        self.task_description = None
        self.task_description_embedding = None
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        # self.gripper_is_closed = False
        self.previous_gripper_action = None
        self.gripper_action_repeat = 0
        self.sticky_gripper_num_repeat = 15

        self.policy_setup = policy_setup
        if self.policy_setup == "google_robot":
            self.unnormalize_action = False
            self.unnormalize_action_fxn = None
            self.invert_gripper_action = False
            self.action_rotation_mode = "axis_angle"
        elif self.policy_setup == "widowx_bridge":
            self.unnormalize_action = True
            self.unnormalize_action_fxn = self._unnormalize_action_widowx_bridge
            self.invert_gripper_action = True
            self.action_rotation_mode = "rpy"
        else:
            raise NotImplementedError()

    def init_model(self):
        self.observation = {}
        self.observation["image"] = torch.zeros((224, 224, 3))
        self.observation["natural_language_embedding"] = torch.zeros((512,), dtype=torch.float32)


    @staticmethod
    def _rescale_action_with_bound(
        actions: np.ndarray,
        low: float,
        high: float,
        safety_margin: float = 0.0,
        post_scaling_max: float = 1.0,
        post_scaling_min: float = -1.0,
    ) -> np.ndarray:
        """Formula taken from https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range."""
        resc_actions = (actions - low) / (high - low) * (post_scaling_max - post_scaling_min) + post_scaling_min
        return np.clip(
            resc_actions,
            post_scaling_min + safety_margin,
            post_scaling_max - safety_margin,
        )

    def _unnormalize_action_widowx_bridge(self, action):
        action["world_vector"] = self._rescale_action_with_bound(
            action["world_vector"],
            low=-1.75,
            high=1.75,
            post_scaling_max=0.05,
            post_scaling_min=-0.05,
        )
        action["rotation_delta"] = self._rescale_action_with_bound(
            action["rotation_delta"],
            low=-1.4,
            high=1.4,
            post_scaling_max=0.25,
            post_scaling_min=-0.25,
        )
        return action


        # Run LLM Forward --> returns CausalLMOutputWithPast!
        return generated_feats 
    def _resize_image(self, image) -> torch.Tensor:
        #image = tf.image.resize_with_pad(image, target_width=self.image_width, target_height=self.image_height)
        #image = tf.cast(image, tf.uint8)
        return image

    def _initialize_task_description(self, task_description: Optional[str] = None) -> None:
        if task_description is not None:
            self.task_description = task_description
            self.task_description_embedding =  torch.zeros((512,), dtype=torch.float32)
        else:
            self.task_description = ""
            self.task_description_embedding = torch.zeros((512,), dtype=torch.float32)

    def reset(self, task_description: str) -> None:
       # self._initialize_model()i
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        # self.gripper_is_closed = False
        self.previous_gripper_action = None
        self._initialize_task_description(task_description)

    @staticmethod
    def _small_action__filter_google_robot(raw_action, arm_movement: bool = False, gripper: bool = True) -> dict:
        # small action filtering for google robot
        if arm_movement:
            raw_action["world_vector"] = torch.where(
                torch.abs(raw_action["world_vector"]) < 5e-3,
                torch.zeros_like(raw_action["world_vector"]),
                raw_action["world_vector"],
            )
            raw_action["rotation_delta"] = torch.where(
                torch.abs(raw_action["rotation_delta"]) < 5e-3,
                torch.zeros_like(raw_action["rotation_delta"]),
                raw_action["rotation_delta"],
            )
            raw_action["base_displacement_vector"] = torch.where(
                raw_action["base_displacement_vector"] < 5e-3,
                torch.zeros_like(raw_action["base_displacement_vector"]),
                raw_action["base_displacement_vector"],
            )
            raw_action["base_displacement_vertical_rotation"] = torch.where(
                raw_action["base_displacement_vertical_rotation"] < 1e-2,
                torch.zeros_like(raw_action["base_displacement_vertical_rotation"]),
                raw_action["base_displacement_vertical_rotation"],
            )
        if gripper:
            raw_action["gripper_closedness_action"] = torch.where(
                torch.abs(raw_action["gripper_closedness_action"]) < 1e-2,
                torch.zeros_like(raw_action["gripper_closedness_action"]),
                raw_action["gripper_closedness_action"],
            )
        return raw_action

    def save_feat(self,exp_name):
        import json
        with open(f'{exp_name}.json', 'w') as f:
            json.dump(self.feats, f)
        with open(f'{exp_name}_vlm.json', 'w') as f:
            json.dump(self.vlm_features, f)
        
    def save_features(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        output_attentions = output_attentions if output_attentions is not None else self.policy.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.policy.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.policy.config.use_return_dict

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        
        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None

        # Note :: We only support forward passes with the following cases:
        #   => Cached Generation :: (input_ids.shape[1] == 1) and (past_key_values is not None)
        #   => Unimodal Forward :: (pixel_values is None)
        #   => Multimodal Forward :: (pixel_values is not None) and (input_ids/embeds.shape[0] == pixel_values.shape[0])

        # === Handle Generation with Cache (`input_ids.shape[1] == 1`) =>> requires `past_keys_values` ===
        if input_ids.shape[1] == 1:
            assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"
            assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
            assert labels is None, "Unexpected key `labels` provided during cached generation!"

            language_model_output = self.policy.language_model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Unimodal Forward ===
        elif pixel_values is None:
            assert (input_ids is not None) and (inputs_embeds is None), "Missing `input_ids` in language-only forward!"
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            language_model_output = self.policy.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Multimodal Forward ===
        elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            # Visual Feature Extraction
            patch_features = self.policy.vision_backbone(pixel_values)

            # Projection Logic =>> Update Attention Mask
            projected_patch_embeddings = self.policy.projector(patch_features)
            projected_patch_attention_mask = None
            if attention_mask is not None:
                projected_patch_attention_mask = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

            # Get Input Embeddings (from Language Model Embeddings)
            input_embeddings = self.policy.get_input_embeddings()(input_ids)

            # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)
            multimodal_embeddings = torch.cat(
                [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
            )
            multimodal_attention_mask = None
            if attention_mask is not None:
                multimodal_attention_mask = torch.cat(
                    [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
                )

            # Build Labels (if specified) =>> Ignore Labels for Patch Embeddings
            multimodal_labels = None
            if labels is not None:
                projected_patch_labels = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=IGNORE_INDEX,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)

            # Dispatch to Language Model
            language_model_output = self.policy.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=multimodal_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            feat_dict = {}
            feat_dict["encoded_feat"] = multimodal_embeddings.mean(1).detach().cpu().to(dtype= torch.float32).numpy().tolist()
            feat_dict["decoded_feat"] = language_model_output.past_key_values[-1][-1].mean(1).mean(1).detach().cpu().to(dtype= torch.float32).numpy().tolist()
            self.feats.append(feat_dict)


        # === Otherwise =>> Assume Invalid! ===
        elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
            raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")

        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output

        return language_model_output

    def str_sample(self,
         inputs, **kwargs: str
    ) -> np.ndarray:
        
        # print(kwargs)
        input_ids = torch.cat(
              (inputs["input_ids"], torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(inputs["input_ids"].device)), dim=1)


        # Run VLA inference
        generated_ids = self.policy.generate(input_ids, max_new_tokens=200, **kwargs)
        return generated_ids
        
        
        
    def normalize_act(self, normalized_actions, unnorm_key):
        action_norm_stats = self.policy.get_action_stats(unnorm_key)

        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))

        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(mask,0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low, normalized_actions)
        return actions
    
    def step(self, image: np.ndarray, task_description = None):
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                # task description has changed; update language embedding
                # self._initialize_task_description(task_description)
                self.reset(task_description)
        
        assert image.dtype == np.uint8
        self.init_model()
        #image = self._resize_image(image)
    
        print()
        prompt ="In: What action should the robot take to {}?\nOut:".format(task_description)
        self.observation["image"] = image
        self.observation["natural_language_embedding"] = self.task_description_embedding
        
        #self.unnorm_key = "bridge_dataset"
        #Use if not using string tokenizer
        # _actions = self.policy.predict_action(Image.fromarray(image), task_description, unnorm_key=self.unnorm_key, do_sample=False)
        #String tokenizer
    
        inputs = self.input_processor(prompt, image).to('cuda:0', dtype=torch.bfloat16)
        flag = True
        while flag:
            text_ids = self.str_sample(inputs, do_sample=True)
            text_answer = text_ids #t_tokenizer.batch_decode(text_ids)[0]
            if len(text_answer.split(','))>=7:
                flag=False
            
        actions = []
        
        try:
            for t in text_answer.split(',')[:7]:
                actions.append(float(t))
            _actions = self.normalize_act(np.array(actions), "bridge_dataset")
        except Exception as e:
            _actions = self.normalize_act(np.array([0,0,0,0,0,0,0]), "bridge_dataset")
        
        #a= self.save_features(**inputs)
        raw_action = {
            "world_vector": np.array(_actions[:3]),
            "rotation_delta": np.array(_actions[3:6]),
            "open_gripper": np.array(_actions[6:7]),  # range [0, 1]; 1 = open; 0 = close
        }
        action = {}
        action["world_vector"] = raw_action["world_vector"] 
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle
        
        raw_action = {
                "world_vector": np.array(_actions[:3]),
            "rotation_delta": np.array(_actions[3:6]),
            "open_gripper": np.array(_actions[6:7]),  # range [0, 1]; 1 = open; 0 = close
        }
        
        action = {}
        action["world_vector"] = raw_action["world_vector"] 
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle
        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]

            # alternative implementation
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = (
                    self.previous_gripper_action - current_gripper_action
                )  # google robot 1 = close; -1 = open
            print("relative_",relative_gripper_action,current_gripper_action,relative_gripper_action)
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                print("second last",relative_gripper_action)
            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

                print("last",relative_gripper_action)
            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            print(relative_gripper_action)
            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = (
                2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            )  # binarize gripper action to 1 (open) and -1 (close)
            # self.gripper_is_closed = (action['gripper'] < 0.0)
        
        return _actions, action

