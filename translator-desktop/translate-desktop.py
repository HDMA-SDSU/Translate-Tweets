from gooey import Gooey, GooeyParser
import pandas as pd
import requests
from tqdm import tqdm
from contextlib import redirect_stderr
import io
import sys
import numpy as np

@Gooey(program_name="COVID-Crowdfight Translate Twitter Data", progress_regex=r"(\d+)%")
def parse_args():
    prog_descrip = "Translate Italian Twitter data to English using DeepL"

    parser = GooeyParser(description=prog_descrip)

    parser.add_argument(
        "file_type",
        metavar="File Type",
        help="Select the source data type (csv/xlsx supported)",
        widget="Dropdown",
        choices=["csv", "xlsx"],
        default="xlsx",
    )

    parser.add_argument(
        "datafile",
        metavar="Twitter Data",
        help="Select the source twitter data file",
        widget="FileChooser",
    )

    parser.add_argument(
        "apifile",
        metavar="API Key Text File",
        help="Select the API Key text file",
        widget="FileChooser",
    )

    parser.add_argument(
        "outfile",
        metavar="Output Data",
        help="Output translated data file (must have .xlsx extension)",
        widget="FileSaver",
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


def translate_series(data,api_key,src_lang="IT",target_lang="EN"):
    # Create empty list
    translated_list = []

    # Translate all tweets and add to list
    # "text" parameter can be an array however there is a limit which is not defined in the documentation    
    parameters = {
        "text": data,
        "source_lang": src_lang,
        "target_lang": target_lang,
        "auth_key": api_key,
    }
    response = requests.get(
        "https://api.deepl.com/v2/translate", params=parameters
    )
    data = response.json()
    for item in data.values():
        for key in item:
            translated_list.append(key["text"])

    return translated_list


if __name__ == "__main__":
    # Initialise GUI
    conf = parse_args()

    # Read first line in text file for API Key
    with open(conf.apifile) as f:
        api_key = f.readline()
    print(f"API Key Read Successful: {api_key}")

    # Check output file type
    if conf.outfile[-5:] != ".xlsx":
        raise ValueError(
            f"Output file must be .xlsx format, {conf.outfile[-5:]} provided"
        )

    # Read input data
    if conf.file_type == "csv":
        df = pd.read_csv(conf.datafile)
    elif conf.file_type == "xlsx":
        df = pd.read_excel(conf.datafile)
    else:
        raise NotImplementedError("File type not supported")

    # Check selected column in data source
    if conf.datacol not in df.columns:
        raise ValueError(
            f"Data column provided: {conf.datacol} not in source data columns {conf.datafile}"
        )

    print(f"Successfully read data from: {conf.datafile}")

    print("Beginning Translation..")

    print("Note: Translation is done is chunks of 40 rows")

    # This is only for showing progress bar
    progress_bar_output = io.StringIO()

    translated_data = []
    with redirect_stderr(progress_bar_output):
        chunk_size = 40
        for _, chunk in tqdm(df.groupby(np.arange(len(df))//chunk_size),file=sys.stdout):

            # Add new data to list
            translated_data.extend(translate_series(chunk[conf.datacol],api_key,conf.srclang,conf.tgtlang))

            print(progress_bar_output.read())

    df[conf.outcol] = translated_data

    # # Translate data
    # translated_df = translate_dataframe(
    #     df, conf.datacol, api_key, conf.srclang, conf.tgtlang
    # )

    # Write to file
    df.to_excel(conf.outfile)

    # DONE!
    print(f"Successfully translated, output file at {conf.outfile}")
