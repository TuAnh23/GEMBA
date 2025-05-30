import os
import sys
import ipdb
import pandas as pd
import diskcache as dc
from absl import app, flags
from gemba.utils import get_gemba_scores_multicand


flags.DEFINE_string('method', "GEMBA-DA-POLYCAND", 'Which method to use?')
flags.DEFINE_string('model', "llama3.3-70B-4bit", 'OpenAI model')
flags.DEFINE_string('data_path', None, 'Filepath to the csv file.')
flags.DEFINE_string('cache_root_dir', "cache", 'Path to the cache directory.')
flags.DEFINE_string('out_full_path', None, 'Filepath to the full LLM output.')
flags.DEFINE_string('out_score_path', None, 'Filepath to the output scores.')
flags.DEFINE_integer('additional_translation_in', 0, 'Additional translations to include as input.')
flags.DEFINE_integer('additional_score_in', 0, 'Additional scores to include as input.')
flags.DEFINE_integer('additional_score_out', 0, 'Additional scores to include as output.')
flags.DEFINE_boolean('use_ref', False, 'Whether to use reference translations.')

def main(argv):
    FLAGS = flags.FLAGS
    out = get_gemba_scores_multicand(
            df=pd.read_csv(FLAGS.data_path), method=FLAGS.method, model=FLAGS.model,
            additional_translation_in=FLAGS.additional_translation_in,
            additional_score_in=FLAGS.additional_score_in,
            additional_score_out=FLAGS.additional_score_out,
            use_ref=FLAGS.use_ref,
            cache_root_dir=FLAGS.cache_root_dir,
    )
    out = pd.DataFrame(out)
    out.to_csv(FLAGS.out_full_path)

    answers = out['answer']

    with open(FLAGS.out_score_path, "w") as file:
        for a in answers:
            file.write(f"{a}\n")


if __name__ == "__main__":
    app.run(main)
