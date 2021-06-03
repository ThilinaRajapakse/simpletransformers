import logging

from simpletransformers.language_generation import LanguageGenerationModel

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model = LanguageGenerationModel(
    "gpt2", "outputs/from_scratch/", args={"max_length": 200}
)
# model = LanguageGenerationModel("gpt2", "outputs/fine-tuned", args={"max_length": 200})
# model = LanguageGenerationModel("gpt2", "gpt2", args={"max_length": 200})

prompts = [
    "Despite the recent successes of deep learning, such models are still far from some human abilities like learning from few examples, reasoning and explaining decisions. In this paper, we focus on organ annotation in medical images and we introduce a reasoning framework that is based on learning fuzzy relations on a small dataset for generating explanations.",
    "There is a growing interest and literature on intrinsic motivations and open-ended learning in both cognitive robotics and machine learning on one side, and in psychology and neuroscience on the other. This paper aims to review some relevant contributions from the two literature threads and to draw links between them.",
    "Recent success of pre-trained language models (LMs) has spurred widespread interest in the language capabilities that they possess. However, efforts to understand whether LM representations are useful for symbolic reasoning tasks have been limited and scattered.",
    "Many theories, based on neuroscientific and psychological empirical evidence and on computational concepts, have been elaborated to explain the emergence of consciousness in the central nervous system. These theories propose key fundamental mechanisms to explain consciousness, but they only partially connect such mechanisms to the possible functional and adaptive role of consciousness.",
    "I failed the first quarter of a class in middle school, so I made a fake report card. I did this every quarter that year. I forgot that they mail home the end-of-year cards, and my mom got it before I could intercept with my fake. She was PISSED—at the school for their error.",
]

for prompt in prompts:
    # Generate text using the model. Verbose set to False to prevent logging generated sequences.
    generated = model.generate(prompt, verbose=False)

    generated = ".".join(generated[0].split(".")[:-1]) + "."
    print(
        "============================================================================="
    )
    print(generated)
    print(
        "============================================================================="
    )
