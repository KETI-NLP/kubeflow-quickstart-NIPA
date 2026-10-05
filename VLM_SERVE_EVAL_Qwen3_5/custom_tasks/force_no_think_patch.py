import lighteval.tasks.prompt_manager
import logging

def patch_tokenizer_in_prompt_manager():
    logger = logging.getLogger(__name__)
    
    old_prepare_chat_template = lighteval.tasks.prompt_manager.PromptManager._prepare_chat_template
    
    def new_prepare_chat_template(self, doc, tokenize=True):
        messages = []
        instruction_used = False

        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})

        for ix, fewshot_sample in enumerate(doc.fewshot_samples):
            query = self._extract_query(fewshot_sample.query, fewshot_sample.instruction)
            if ix == 0 and doc.instruction is not None:
                instruction_used = True
                query = doc.instruction + query
            messages.append({"role": "user", "content": query})
            messages.append({"role": "assistant", "content": fewshot_sample.get_golds()[0]})

        main_query = self._extract_query(doc.query, doc.instruction)
        if doc.instruction is not None and not instruction_used:
            main_query = doc.instruction + main_query
            
        messages.append({"role": "user", "content": main_query})

        if tokenize:
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        else:
            return messages

    lighteval.tasks.prompt_manager.PromptManager._prepare_chat_template = new_prepare_chat_template
    logger.info("Successfully patched PromptManager to pass enable_thinking=False")

def patch_tokenizer_in_prompt_manager_multimodal():
    logger = logging.getLogger(__name__)
    
    def new_prepare_prompt_multimodal(self, doc):
        if self.use_chat_template is False or self.tokenizer is None:
            raise ValueError("Multimodal prompts are only supported with chat template format.")

        if doc.images is None:
            raise ValueError("Multimodal prompts require images to be provided in the document.")

        text_content = [{"type": "text", "text": doc.query}]
        image_content = [{"type": "image", "image": image} for image in doc.images]
        message = {"role": "user", "content": text_content + image_content}

        system_prompt = self.system_prompt or ""
        instruction = doc.instruction or ""
        if system_prompt or instruction:
            system_content = [{"type": "text", "text": system_prompt + instruction}]
            system_prompt_message = {"role": "system", "content": system_content}
            message = [system_prompt_message, message]
        else:
            message = [message]

        try:
            return self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
            
    lighteval.tasks.prompt_manager.PromptManager.prepare_prompt_multimodal = new_prepare_prompt_multimodal
    logger.info("Successfully patched PromptManager multimodal to pass enable_thinking=False")

patch_tokenizer_in_prompt_manager()
patch_tokenizer_in_prompt_manager_multimodal()
