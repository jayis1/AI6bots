import logging
import os
import time
import requests
import discord
from discord.ext import commands
from transformers import AutoTokenizer, AutoModelForCausalLM # Import only necessary classes
import torch

class AIModelManager:
    """
    A class to handle the loading and operation of a local text generation AI model.
    """
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = 'cpu'
        self.load_model()

    def load_model(self):
        """Loads the default text generation model."""
        logging.info("Attempting to load text generation AI model...")

        # Load a text generation model (e.g., gpt2)
        # Replace "gpt2" with the model name you want to use (e.g., a MiniGPT variant if compatible)
        try:
            model_name = "gpt2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

            if torch.cuda.is_available():
                 self.model.to('cuda')
                 self.device = 'gpu'
                 logging.info(f"Successfully loaded text generation model '{model_name}' on GPU.")
            else:
                 self.device = 'cpu'
                 logging.info(f"Successfully loaded text generation model '{model_name}' on CPU.")

            # Add a padding token if the tokenizer doesn't have one (common for GPT-like models)
            if self.tokenizer.pad_token is None:
                 self.tokenizer.pad_token = self.tokenizer.eos_token
                 self.model.config.pad_token_id = self.model.config.eos_token_id


        except Exception as e:
            logging.error(f"Failed to load text generation model '{model_name}': {e}", exc_info=True)
            self.model = None
            self.tokenizer = None


        logging.info(f"Finished loading model. Model loaded: {'gpt2' if self.model else 'None'}")


    def process_command(self, command_name, *args):
        """
        Processes a command using the loaded text generation model.
        Crafts prompts based on the command.
        """
        if not self.model or not self.tokenizer:
            return "AI model is not loaded."

        prompt = ""
        max_length = 150 # Default max length for generated text

        if command_name == 'ask':
            question = args[0] if args else ""
            context = args[1] if len(args) > 1 else ""
            if context:
                 prompt = f"Answer the following question based on the context.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
                 max_length = 200 # Allow longer answers with context
            else:
                 prompt = f"Answer the following question: {question}\n\nAnswer:"
                 max_length = 150

        elif command_name == 'summarize':
             text = args[0] if args else ""
             prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
             max_length = 100 # Summaries should be concise

        elif command_name == 'jokeplease':
            prompt = "Tell me a short, funny joke:\n\nJoke:"
            max_length = 80 # Jokes should be short

        else:
            return "Unknown AI command."

        logging.info(f"Generating text for command '{command_name}' with prompt: '{prompt[:100]}...'")
        start_time = time.time()

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate text
            output_sequences = self.model.generate(
                inputs['input_ids'],
                max_length=max_length + len(inputs['input_ids'][0]), # max_length is total length
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id # Use pad_token_id for generation
            )

            end_time = time.time()
            inference_time = end_time - start_time
            logging.info(f"Text generation inference time: {inference_time:.4f} seconds.")


            generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)

            # Clean up the generated text (remove the prompt)
            if generated_text.startswith(prompt):
                 generated_text = generated_text[len(prompt):].strip()

            return generated_text if generated_text else "Could not generate a response."

        except Exception as e:
            logging.error(f"Error during '{command_name}' command processing: {e}", exc_info=True)
            return "An error occurred while processing your request."


    def is_ready(self):
        """Check if the model is loaded."""
        return self.model is not None and self.tokenizer is not None


class NeuralNetworkCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.ai_manager = AIModelManager()

    @commands.command(name='ask', help='Ask the AI a question. Optionally provide context after the question.')
    async def ask(self, ctx, *, args):
        """Asks the AI a question."""
        # Split args into question and optional context
        parts = args.split('|', 1)
        question = parts[0].strip()
        context = parts[1].strip() if len(parts) > 1 else ""

        if not self.ai_manager.is_ready():
            await ctx.send("AI model is not loaded yet. Please try again later.")
            return

        await ctx.send("Thinking...")
        response = self.ai_manager.process_command('ask', question, context)
        await ctx.send(response)

    @commands.command(name='summarize', help='Summarize the provided text.')
    async def summarize(self, ctx, *, text):
        """Summarizes the provided text."""
        if not self.ai_manager.is_ready():
            await ctx.send("AI model is not loaded yet. Please try again later.")
            return

        await ctx.send("Summarizing...")
        response = self.ai_manager.process_command('summarize', text)
        await ctx.send(response)

    @commands.command(name='jokeplease', help='Ask the AI for a joke.')
    async def jokeplease(self, ctx):
        """Asks the AI for a joke."""
        if not self.ai_manager.is_ready():
            await ctx.send("AI model is not loaded yet. Please try again later.")
            return

        await ctx.send("Attempting to generate a joke...")
        response = self.ai_manager.process_command('jokeplease')
        await ctx.send(response)


async def setup(bot):
    await bot.add_cog(NeuralNetworkCog(bot))
