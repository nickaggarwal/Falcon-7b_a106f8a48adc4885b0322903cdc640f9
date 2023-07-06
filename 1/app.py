import json
import numpy as np
import torch
from transformers import pipeline


class InferlessPythonModel:
    def initialize(self):
        self.generator = pipeline(
            "text-generation",
            model="tiiuae/falcon-7b-instruct",
            device=0,
            use_auth_token="False",
        )

    def infer(self, prompt):
        pipeline_output = self.generator(prompt, do_sample=True, min_length=20)
        generated_txt = pipeline_output[0]["generated_text"]
        return generated_txt

    def finalize(self):
        self.pipe = None
