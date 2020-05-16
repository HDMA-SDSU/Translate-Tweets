from gooey import Gooey, GooeyParser
import pandas as pd
import requests
from tqdm import tqdm
from contextlib import redirect_stderr
import io
import sys
import numpy as np
import json
import os
from joblib import Parallel, delayed


@Gooey(program_name="COVID-Crowdfight Translate Twitter Data", progress_regex=r"(\d+)%")
def parse_args():
    prog_descrip = "Translate Italian Twitter data to English using DeepL"

    parser = GooeyParser(description=prog_descrip)

    parser.add_argument(
        "datadir",
        metavar="Twitter Data",
        help="Select the source twitter data file",
        widget="DirChooser",
    )

    parser.add_argument(
        "apifile",
        metavar="API Key Text File",
        help="Select the API Key text file",
        widget="FileChooser",
    )

    parser.add_argument(
        "--srclang",
        metavar="Source Language",
        help="Select language to translate from",
        choices=deepl_supported_langs(),
        default="IT",
        widget="Dropdown",
    )

    parser.add_argument(
        "--tgtlang",
        metavar="Target Language",
        help="Select language to translate to",
        choices=deepl_supported_langs(),
        default="EN",
        widget="Dropdown",
    )

    parser.add_argument(
        "--datacol",
        metavar="Data Column",
        help="Name of column with data to translate",
        type=str,
        default="full_text",
    )

    parser.add_argument(
        "--outcol",
        metavar="Output Data Column Name",
        help="Name of column to place translated data",
        type=str,
        default="translated_full_text",
    )

    return parser.parse_args()


def deepl_supported_langs():
    # As per https://www.deepl.com/docs-api/translating-text/request/
    # "EN" - English
    # "FR" - French
    # "IT" - Italian
    # "JA" - Japanese
    # "ES" - Spanish
    # "NL" - Dutch
    # "PL" - Polish
    # "PT" - Portuguese (all Portuguese varieties mixed)
    # "RU" - Russian
    # "ZH" - Chinese

    return ["EN", "FR", "IT", "JA", "ES", "NL", "PL", "PT", "RU", "ZH"]


def translate_series(data, api_key, src_lang="IT", target_lang="EN"):
    # Create empty list
    translated_list = []

    try:
        # Translate all tweets and add to list
        # "text" parameter can be an array however there is a limit which is not defined in the documentation
        parameters = {
            "text": data,
            "source_lang": src_lang,
            "target_lang": target_lang,
            "auth_key": api_key,
        }
        response = requests.get("https://api.deepl.com/v2/translate", params=parameters)
        deepl_response_data = response.json()
        for item in deepl_response_data.values():
            for key in item:
                translated_list.append(key["text"])
    except json.decoder.JSONDecodeError:
        # Insert error for each line in data

        for _ in data:
            translated_list.append("Error")
        print(f"Error translating.. `Error` placed in output dataset")

    return translated_list

def translate_file(directory, filename):
    print(os.path.join(directory, filename))

    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(directory, filename))
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(os.path.join(directory, filename))
    else:
        raise NotImplementedError("File type not supported")

    # Check selected column in data source
    if conf.datacol not in df.columns:
        raise ValueError(
            f"Data column provided: {conf.datacol} not in source data columns {filename}"
        )

    print(f"Successfully read data from: {filename}")

    print("Beginning Translation..")

    print("Note: Translation is done is chunks of 35 rows")

    # This is only for showing progress bar
    progress_bar_output = io.StringIO()

    translated_data = []
    with redirect_stderr(progress_bar_output):

        # DeepL supports chunks of 35 items to translate at a time
        chunk_size = 35
        for _, chunk in tqdm(
            df.groupby(np.arange(len(df)) // chunk_size), file=sys.stdout
        ):

            # Add new data to list
            translated_data.extend(
                translate_series(
                    chunk[conf.datacol], api_key, conf.srclang, conf.tgtlang
                )
            )

            print(progress_bar_output.read())

    df[conf.outcol] = translated_data

    # Write to file
    df.to_excel(os.path.join(translated_dir, filename+".xlsx"))
    df.to_pickle(os.path.join(translated_dir, filename+".pkl"))

    # DONE!
    print(f"Successfully translated, output file at {filename}")

    return

if __name__ == "__main__":
    # Initialise GUI
    conf = parse_args()

    # Read first line in text file for API Key
    with open(conf.apifile) as f:
        api_key = f.readline()
    print(f"API Key Read Successful: {api_key}")

    # Read input data
    directory = conf.datadir

    # Prepare a directory for translated texts
    translated_dir = os.path.join(directory, "translated")
    os.mkdir(translated_dir)

    Parallel(n_jobs=8)(delayed(translate_file)(directory, filename) for filename in os.listdir(directory))
