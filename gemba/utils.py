import ipdb
import pandas as pd
import diskcache as dc
from gemba.gpt_api import GptApi
from gemba.gemba_mqm_utils import TEMPLATE_GEMBA_MQM, apply_template, parse_mqm_answer
from gemba.gemba_esa import TEMPLATE_GEMBA_ESA_ERROR_SPANS, TEMPLATE_GEMBA_ESA_RANKING
from gemba.prompt import prompts, validate_number, create_polycand_prompt, create_polyic_prompt
import asyncio


def get_gemba_scores(source, hypothesis, source_lang, target_lang, method, model):
    df = pd.DataFrame({'source_seg': source, 'target_seg': hypothesis})
    df['source_lang'] = source_lang
    df['target_lang'] = target_lang

    cache = dc.Cache(f'cache/{model}_{method}', expire=None, size_limit=int(10e10), cull_limit=0, eviction_policy='none')
    gptapi = GptApi()

    if method == "GEMBA-MQM":
        df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_MQM, x), axis=1)
        parse_answer = lambda x: parse_mqm_answer(x, list_mqm_errors=False, full_desc=True)
        answers = gptapi.bulk_request(df, model, parse_answer, cache=cache, max_tokens=500)
    elif method in ["GEMBA-DA", "GEMBA-DA_ref", "GEMBA-SQM", "GEMBA-SQM_ref", "GEMBA-stars", "GEMBA-stars_ref", "GEMBA-classes", "GEMBA-classes_ref"]:
        df["prompt"] = df.apply(lambda x: apply_template(prompts[method]['prompt'], x), axis=1)
        parse_answer = prompts[method]["validate_answer"]
        answers = gptapi.bulk_request(df, model, parse_answer, cache=cache, max_tokens=500)
    elif method == "GEMBA-ESA":
        df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_ESA_ERROR_SPANS, x), axis=1)
        parse_answer = lambda x: x
        error_spans = gptapi.bulk_request(df, model, parse_answer, cache=cache)
        df['error_spans'] = pd.DataFrame(error_spans)['answer']

        df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_ESA_RANKING, x), axis=1)
        parse_answer = validate_number
        answers = gptapi.bulk_request(df, model, parse_answer, cache=cache)
    else:
        raise Exception(f"Method {method} not supported.")

    return list(pd.DataFrame(answers)['answer'])


def get_gemba_scores_polycand(
        df, method, model,
        additional_translation_in: int = 0,
        additional_score_in: int = 0,
        additional_score_out: int = 0,
        use_ref: bool = False,
        cache_root_dir: str = "cache"
):
    """
    Args:
        df: Dataframe with columns [langs,src,ref,mt,score,mt2,score2,mt3,score3,mt4,score4,mt5,score5,mt6,score6]
    """

    assert method == "GEMBA-DA-POLYCAND"

    df["prompt"] = df.apply(
        lambda x: create_polycand_prompt(
            data=x, additional_score_in=additional_score_in,
            additional_score_out=additional_score_out,
            additional_translation_in=additional_translation_in,
            use_ref=use_ref),
        axis=1
    )

    cache = dc.Cache(
        f'{cache_root_dir}/{model}_{method}_{additional_translation_in}_{additional_score_in}_{additional_score_out}_{use_ref}',
        expire=None, size_limit=int(10e10), cull_limit=0,
        eviction_policy='none'
    )
    gptapi = GptApi()
    parse_answer = prompts[method]["validate_answer"]
    answers = asyncio.run(gptapi.bulk_request(df, model, parse_answer, cache=cache, max_tokens=500))

    return answers


def get_gemba_scores_polyic(
        df, method, model,
        additional_sample_in: int = 0,
        use_ref: bool = False,
        cache_root_dir: str = "cache"
):
    """
    Args:
        df: Dataframe with columns [langs,src,ref,mt,score,src2,mt2,score2,src3,mt3,score3,src4,mt4,score4,src5,mt5,score5,src6,mt6,score6]
    """

    assert method == "GEMBA-DA-POLYIC"

    df["prompt"] = df.apply(
        lambda x: create_polyic_prompt(
            data=x, additional_sample_in=additional_sample_in,
            use_ref=use_ref),
        axis=1
    )

    cache = dc.Cache(
        f'{cache_root_dir}/{model}_{method}_{additional_sample_in}_{use_ref}',
        expire=None, size_limit=int(10e10), cull_limit=0,
        eviction_policy='none'
    )
    gptapi = GptApi()
    parse_answer = prompts[method]["validate_answer"]
    answers = asyncio.run(gptapi.bulk_request(df, model, parse_answer, cache=cache, max_tokens=500))

    return answers
