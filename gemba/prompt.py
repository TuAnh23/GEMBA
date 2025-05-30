import re
from termcolor import colored
import pandas as pd


def parse_and_check_numerical_answer(answer, min=None, max=None):
    attempt = parse_numerical_answer(answer, min, max)
    if attempt is not None:
        if attempt < min or attempt > max:
            return None
        return attempt

    return None


def parse_numerical_answer(answer, min=None, max=None):
    # get all numbers in a string
    numbers = re.findall(r'\d+', answer)
    if len(numbers) == 1:
        return int(numbers[0])

    # check if the answer is in form ['100'] and extract the number
    r1 = re.match(r"^\[['\"][0-9]*['\"]\]$", answer)
    if r1 is not None:
        return int(answer[2:-2])

    if max is not None:
        # check if the answer is in a form of 0/100
        r2 = re.match(rf"^[0-9]*/{max}$", answer)
        if r2 is not None:
            return int(answer.split("/")[0])

    return None


def validate_number(x, min=0, max=100):
    attempt = parse_and_check_numerical_answer(x, min, max)
    if attempt is not None:
        return attempt
    return None


def validate_number_multicand(x, min=0, max=100):
    # Remove the "/100" substring in case the answer is in the form 0/100
    x = x.replace(f'/{max}', '')

    # Get all numerical values (both integers and floats) in a string
    numbers = re.findall(r'\d+\.\d+|\d+', x)

    if len(numbers) == 0:
        return None

    # Consider the last number
    result_number = float(numbers[-1])

    # Check if the number is out of range
    if result_number < min or result_number > max:
        return None
    return result_number


def parse_classes(answer, classes):
    final_class = None
    for i in range(len(classes)):
        if classes[i].lower() in answer.lower():
            if final_class is None:
                final_class = i
            else:
                print(colored(f"Two classes found in answer {answer}", "red"))
                return None

    return final_class


def validate_stars(x):
    x = x.lower()
    # try to find all possible answers as sometimes it seems to be explaining itself
    possible_answers = set()

    # check if string x contains * characters
    if "*" in x:
        possible_answers.add(x.count("*"))
    if "★" in x:
        possible_answers.add(x.count("★"))

    x = f" {x} ".replace("\n", " ")
    # possible answers: "five stars", "5 stars", "five", "five starts: perfect translation", ...
    if " one " in x or "1 star" in x:
        possible_answers.add(1)
    if " two " in x or "2 star" in x:
        possible_answers.add(2)
    if " three " in x or "3 star" in x:
        possible_answers.add(3)
    if " four " in x or "4 star" in x:
        possible_answers.add(4)
    if " five " in x or "5 star" in x:
        possible_answers.add(5)

    numerical = parse_numerical_answer(x)
    if numerical is not None:
        possible_answers.add(numerical)

    if len(possible_answers) == 1:
        answer = possible_answers.pop()
        if 1 <= answer <= 5:
            return answer
    return None


language_codes = {
    "ja": "Japanese",
    "de": "German",
    "is": "Icelandic",
    "ru": "Russian",
    "es": "Spanish",
    "en": "English",
    "uk": "Ukrainian",
    "zh": "Chinese",
    "cs": "Czech",
    "hi": "Hindi"
}

prompts = {
    "GEMBA-DA": {
        "prompt": 'Score the following translation from {source_lang} to {target_lang} on a continuous scale from 0 to 100, where a score of zero means "no meaning preserved" and score of one hundred means "perfect meaning and grammar".\n\n{source_lang} source: "{source_seg}"\n{target_lang} translation: "{target_seg}"\nScore: ',
        "validate_answer": lambda x: validate_number(x),
        "use_ref": False},

    "GEMBA-DA_ref": {
        "prompt": 'Score the following translation from {source_lang} to {target_lang} with respect to human reference on a continuous scale 0 to 100 where score of zero means "no meaning preserved" and score of one hundred means "perfect meaning and grammar".\n\n{source_lang} source: "{source_seg}"\n{target_lang} human reference: {reference_seg}\n{target_lang} machine translation: "{target_seg}"\nScore: ',
        "validate_answer": lambda x: validate_number(x),
        "use_ref": True},

    "GEMBA-SQM": {
        "prompt": 'Score the following translation from {source_lang} to {target_lang} on a continuous scale from 0 to 100 that starts on "No meaning preserved", goes through "Some meaning preserved", then "Most meaning preserved and few grammar mistakes", up to "Perfect meaning and grammar".\n\n{source_lang} source: "{source_seg}"\n{target_lang} translation: "{target_seg}"\nScore (0-100): ',
        "validate_answer": lambda x: validate_number(x),
        "use_ref": False},

    "GEMBA-SQM_ref": {
        "prompt": 'Score the following machine translation from {source_lang} to {target_lang} with respect to the human reference on a continuous scale from 0 to 100 that starts with "No meaning preserved", goes through "Some meaning preserved", then "Most meaning preserved and few grammar mistakes", up to "Perfect meaning and grammar".\n\n{source_lang} source: "{source_seg}"\n{target_lang} human reference: "{reference_seg}"\n{target_lang} machine translation: "{target_seg}"\nScore (0-100): ',
        "validate_answer": lambda x: validate_number(x),
        "use_ref": True},

    "GEMBA-stars": {
        "prompt": 'Score the following translation from {source_lang} to {target_lang} with one to five stars. Where one star means "Nonsense/No meaning preserved", two stars mean "Some meaning preserved, but not understandable", three stars mean "Some meaning preserved and understandable", four stars mean "Most meaning preserved with possibly few grammar mistakes", and five stars mean "Perfect meaning and grammar".\n\n{source_lang} source: "{source_seg}"\n{target_lang} translation: "{target_seg}"\nStars: ',
        "validate_answer": lambda x: validate_stars(x),
        "use_ref": False},

    "GEMBA-stars_ref": {
        "prompt": 'Score the following translation from {source_lang} to {target_lang} with respect to the human reference with one to five stars. Where one star means "Nonsense/No meaning preserved", two stars mean "Some meaning preserved, but not understandable", three stars mean "Some meaning preserved and understandable", four stars mean "Most meaning preserved with possibly few grammar mistakes", and five stars mean "Perfect meaning and grammar".\n\n{source_lang} source: "{source_seg}"\n{target_lang} human reference: "{reference_seg}"\n{target_lang} translation: "{target_seg}"\nStars: ',
        "validate_answer": lambda x: validate_stars(x),
        "use_ref": True},

    "GEMBA-classes": {
        "prompt": 'Classify the quality of machine translation from {source_lang} to {target_lang} into one of following classes: "No meaning preserved", "Some meaning preserved, but not understandable", "Some meaning preserved and understandable", "Most meaning preserved, minor issues", "Perfect translation".\n\n{source_lang} source: "{source_seg}"\n{target_lang} machine translation: "{target_seg}"\nClass: ',
        "use_ref": False,
        "validate_answer": lambda x, classes=["No meaning preserved", "Some meaning preserved, but not understandable", "Some meaning preserved and understandable", "Most meaning preserved, minor issues", "Perfect translation"]: parse_classes(x, classes),
        "max_tokens": 100},

    "GEMBA-classes_ref": {
        "prompt": 'Classify the quality of machine translation from {source_lang} to {target_lang} with respect to the human reference into one of following classes: "No meaning preserved", "Some meaning preserved, but not understandable", "Some meaning preserved and understandable", "Most meaning preserved, minor issues", "Perfect translation".\n\n{source_lang} source: "{source_seg}"\n{target_lang} human reference: "{reference_seg}"\n{target_lang} machine translation: "{target_seg}"\nClass: ',
        "use_ref": True,
        "validate_answer": lambda x, classes=["No meaning preserved", "Some meaning preserved, but not understandable", "Some meaning preserved and understandable", "Most meaning preserved, minor issues", "Perfect translation"]: parse_classes(x, classes),
        "max_tokens": 100},

    "GEMBA-DA-POLYCAND": {
        "validate_answer": lambda x: validate_number_multicand(x)},
}


def create_multicand_prompt(
        data: pd.Series,
        additional_translation_in: int = 0,
        additional_score_in: int = 0,
        additional_score_out: int = 0,
        use_ref: bool = False
):
    """

    Args:
        data: datapoint containing [langs,src,ref,mt,score,mt2,score2,mt3,score3,mt4,score4,mt5,score5,mt6,score6]

    Returns:
        prompt

    """
    assert additional_score_in == 0 or additional_score_in == additional_translation_in
    assert additional_score_out == 0 or additional_score_out == additional_translation_in
    assert additional_translation_in <= 5

    source_lang = language_codes[data['langs'].split('/')[-1].split('-')[0]]
    target_lang = language_codes[data['langs'].split('/')[-1].split('-')[1]]

    additional_prompt = ''

    if additional_translation_in > 0 and additional_score_in > 0:
        additional_prompt = '--------------------------------------------------------------\n'
        if additional_translation_in == 1:
            additional_prompt += "Below is an example translation along with its score: \n"
        else:
            additional_prompt += "Below are some example translations along with their scores: \n"

        for i in range(0, additional_translation_in):
            sample_translation = data[f"mt{i+2}"]
            sample_score = data[f"score{i+2}"]
            additional_prompt += f'\n{target_lang} translation: "{sample_translation}"\nScore: {sample_score}\n'

        additional_prompt += '\n--------------------------------------------------------------\n'

    elif additional_translation_in > 0:
        additional_prompt = '--------------------------------------------------------------\n'
        if additional_translation_in == 1:
            additional_prompt += "Below is an example translation: \n"
        else:
            additional_prompt += "Below are some example translations: \n"

        for i in range(0, additional_translation_in):
            sample_translation = data[f"mt{i+2}"]
            additional_prompt += f'\n{target_lang} translation: {sample_translation}\n'

        if additional_score_out == 1:
            additional_prompt += "\nFirst, output the score of the above translation. \n"
        elif additional_score_out > 1:
            additional_prompt += "\nFirst, output the scores of the above translations. \n"

        additional_prompt += '--------------------------------------------------------------\n\n'

    if use_ref:
        ref_prompt = f"{target_lang} human reference: {data['ref']}"
    else:
        ref_prompt = ""

    prompt = f'Score the translation provided at the end of this prompt from {source_lang} to {target_lang} ' \
             f'{"with respect to human reference " if use_ref else ""}' \
             f'on a continuous scale from 0 to 100, where a score of zero means "no meaning preserved" ' \
             f'and score of one hundred means "perfect meaning and grammar". ' \
             f'Keep your explanation as short as possible. ' \
             f'Provide the final score at the end of your answer, do not output anything else afterward. \n\n' \
             f'{source_lang} source: {data["src"]}\n' \
             f'{ref_prompt}\n' \
             f'\n' \
             f'{additional_prompt}' \
             f'{"Now" if additional_score_out == 0 else "Then"} score this translation ' \
             f'(remember to output the final score only at the end of your answer):\n' \
             f'{target_lang} translation: {data["mt"]}\nScore: '

    return prompt
