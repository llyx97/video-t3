import json
import os
import random
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# LLAVA PATH: download from https://huggingface.co/datasets/Lin-Chen/Open-LLaVA-NeXT-mix1M
llava_next_path = "open-llava-next_instruct_mix1M.json"

# textual temporal data dir: download from https://huggingface.co/datasets/MMInstruction/Video-T3-QA
textual_tempooral_data_path = "DIR_TO_VIDEO_T3_QA"
# DIR to save the synthesized dataset
save_dir = "DIR_TO_SAVE"
# PATH to hotpot QA, for comparasion
hotpot_qa_path = "DIR_TO_HOTPOT_QA/hotpot_train_v1.1.json"


# Hotpot QA for comparasion
def convert_hotpot_qa(
    hotpotqa_path=hotpot_qa_path,
    limit=-1,
    max_length=-1,
    repeat=False,
):
    hotpotqa = json.load(open(hotpotqa_path, "r"))
    llava_format = []
    human_texts = []
    answer_texts = []
    for instance in hotpotqa:
        q = instance["question"]
        a = instance["answer"]
        context = instance["context"]
        random.shuffle(context)
        human_text = "Given the following paragraphs, answer the question. \n"
        description = ""
        for para in context:
            description += (
                "Title: " + para[0] + "\n" + "Content: " + "".join(para[1]) + "\n\n"
            )
        if repeat:  # repeat the last paragraph title
            human_text += (
                description
                + "\n"
                + "Question: "
                + "What's the title of the last paragraph?"
            )
            a = context[-1][0]
        else:
            human_text += description + "\n" + "Question: " + q + "\n"
        llava_instance = {
            "id": "hotpotqa-" + instance["_id"],
            "conversations": [
                {"from": "human", "value": human_text},
                {"from": "gpt", "value": a},
            ],
        }
        if len(human_text.split()) > max_length:
            # print("Text too long: ", len(human_text.split()))
            continue
        human_texts.append(human_text)
        answer_texts.append(a)
        llava_format.append(llava_instance)
    random.shuffle(llava_format)
    print("Total instances after filtering: ", len(llava_format))
    tokenizer = AutoTokenizer.from_pretrained("lmms-lab/LongVA-7B")
    human_word_lengths = [len(tokenizer.tokenize(text)) for text in human_texts]
    # word_lengths = [len(human_text.split()) for human_text in human_texts]
    answer_word_lengths = [len(tokenizer.tokenize(text)) for text in answer_texts]

    print("average human text length: ", sum(human_word_lengths) / len(human_texts))
    print("average answer text length: ", sum(answer_word_lengths) / len(answer_texts))
    if limit > 0:
        llava_format = llava_format[:limit]
    print("Total instances: ", len(llava_format))
    return llava_format, human_texts, answer_texts


# length check
def plot_text_length_distrbution(
    human_texts, answer_texts, dataset_name, tokenizer=None
):
    # plot text length distribution
    # Calculate word lengths
    if tokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        human_word_lengths = [len(tokenizer.tokenize(text)) for text in human_texts]
        answer_word_lengths = [len(tokenizer.tokenize(text)) for text in answer_texts]
    else:
        human_word_lengths = [len(text.split()) for text in human_texts]
        answer_word_lengths = [len(text.split()) for text in answer_texts]

    # Create the plot with two subfigures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Word Length Distribution - {dataset_name}", fontsize=16)

    # Plot human text distribution
    ax1.hist(human_word_lengths, bins=50, color="blue", alpha=0.7)
    ax1.set_title("Human Text")
    ax1.set_xlabel("Word Count")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Plot answer text distribution
    ax2.hist(answer_word_lengths, bins=50, color="red", alpha=0.7)
    ax2.set_title("Answer Text")
    ax2.set_xlabel("Word Count")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout and save the plot as a PDF
    plt.tight_layout()
    plt.savefig(
        f"word_length_distribution_{dataset_name}.pdf", dpi=300, bbox_inches="tight"
    )

    # Close the plot to free up memory
    plt.close()

    print(f"Plot saved as 'word_length_distribution_{dataset_name}.pdf'")


def load_tempqa_data(
    path="/home/lilei/TempCompass/tempqa/LLaVA-ReCap-558K-Combine-first-sent-qa-train.json",
    aspect="order",
    version="v1",
):

    llava_format = []

    tempqa_data = json.load(open(path, "r"))

    for idx, instance in enumerate(tempqa_data):
        assert len(instance) == 2
        assert instance[0]["from"] == "human"
        assert instance[1]["from"] == "gpt"
        llava_instance = {
            "id": f"tempqa-{aspect}-{version}" + str(idx),
            "conversations": [
                {"from": "human", "value": instance[0]["value"]},
                {"from": "gpt", "value": instance[1]["value"]},
            ],
        }
        llava_format.append(llava_instance)
    return llava_format


if __name__ == "__main__":
    random.seed(1234)

    # Hotpot QA for comparasion if needed
    # hotpot_qa_llava_format, hotpot_qa_human_texts, hotpot_qa_answer_texts = (
    #     convert_hotpot_qa(limit=-1, max_length=16000, repeat=False)
    # )
    data = {}

    # load llava-next data
    with open(llava_next_path, "r") as f:
        data["llava_next"] = json.load(f)
        # data["llava_next"] =  [ d for d in llava_next_data if not no_image(d)]
        random.shuffle(data["llava_next"])

    print("Total llava next data: ", len(data["llava_next"]))


    data_dict = {
        # "caporder_random_train": caporder_random_train,
        # "caporder_gpt_train": caporder_gpt_train,
        "tempqa_v2_order_long_1x": load_tempqa_data(
            aspect="order-long",
            version="v2",
            path=f"{textual_tempooral_data_path}/temp_qa_order_train_long.json",
        ),
        "tempqa_v2_attribute_long_1x": load_tempqa_data(
            aspect="attribute-long",
            version="v2",
            path=f"{textual_tempooral_data_path}/temp_qa_attribute_train_long.json",
        ),
        "tempqa_v2_order_long_2x": load_tempqa_data(
            aspect="order-2x",
            version="v2-long",
            path=f"{textual_tempooral_data_path}/temp_qa_order_train_long_2x_15k.json",
        ),
        "tempqa_v2_attribute_long_2x": load_tempqa_data(
            aspect="attribute-2x",
            version="v2-long",
            path=f"{textual_tempooral_data_path}/temp_qa_attribute_train_long_2x_15k.json",
        ),
        "tempqa_v2_order_long_4x": load_tempqa_data(
            aspect="order-4x",
            version="v2-long",
            path=f"{textual_tempooral_data_path}/temp_qa_order_train_long_4x_15k.json",
        ),
        "tempqa_v2_attribute_long_4x": load_tempqa_data(
            aspect="attribute-4x",
            version="v2-long",
            path="/home/lilei/TempCompass/vript_data/temp_qa_attribute_train_long_4x_15k.json",
        ),
        "tempqa_v2_attribute_long_8x": load_tempqa_data(
            aspect="attribute-8x",
            version="v2-long",
            path=f"{textual_tempooral_data_path}/temp_qa_attribute_train_long_8x_15k.json",
        ),
        "tempqa_v2_order_long_8x": load_tempqa_data(
            aspect="order-8x",
            version="v2-long",
            path=f"{textual_tempooral_data_path}/temp_qa_order_train_long_8x_15k.json",
        ),
        "temp_qa_refer_anchor_temp2any_train": load_tempqa_data(
            aspect="refer_anchor_temp2any",
            version="v2",
            path=f"{textual_tempooral_data_path}/temp_qa_refer_anchor_temp2any_train.json",
        ),
        "temp_qa_refer_begin_end_temp2any_train": load_tempqa_data(
            aspect="refer_begin_end_temp2any",
            version="v2",
            path=f"{textual_tempooral_data_path}/temp_qa_refer_begin_end_temp2any_train.json",
        ),
        "temp_qa_order_shuffle_phrase_train": load_tempqa_data(
            aspect="order_shuffle_phrase_fix",
            version="v2",
            path=f"{textual_tempooral_data_path}/temp_qa_order_shuffle_phrase_train.json",
        ),
        "temp_qa_order_shuffle_prefix_train": load_tempqa_data(
            aspect="order_shuffle_prefix_fix",
            version="v2",
            path=f"{textual_tempooral_data_path}/temp_qa_order_shuffle_prefix_train.json",
        ),
        "temp_qa_order_shuffle_sentence_train": load_tempqa_data(
            aspect="order_shuffle_sentence",
            version="v2",
            path=f"{textual_tempooral_data_path}/temp_qa_order_shuffle_sentence_train.json",
        ),
    }

    data.update(data_dict)

    def no_image(instance):
        return (
            "image" not in instance
            or instance["image"] == ""
            or instance["image"] is None
        )

    # # load llava next data
    for k in data:
        print(k, len(data[k]))
        random.shuffle(data[k])

    text_dataset_limit = 15_000  # len(tempqa_data) + len(hotpot_qa_llava_format)
    llava_next_ratio = 0  # 10 times of the text dataset
    total_limit = 30_000
    # data composition 1
    data_compositions = [
        [
            "temp_qa_order_shuffle_phrase_train",
            "temp_qa_order_shuffle_prefix_train",
            "temp_qa_order_shuffle_sentence_train",
            "temp_qa_refer_anchor_temp2any_train",
            "temp_qa_refer_begin_end_temp2any_train",
        ],
        # # shuffle + order
        [
            "temp_qa_order_shuffle_phrase_train",
            "temp_qa_order_shuffle_prefix_train",
            "temp_qa_order_shuffle_sentence_train",
            "tempqa_v2_order_long_1x",
        ],
        # # shuffle + attribute
        [
            "temp_qa_order_shuffle_phrase_train",
            "temp_qa_order_shuffle_prefix_train",
            "temp_qa_order_shuffle_sentence_train",
            "tempqa_v2_attribute_long_1x",
        ],
        # # shuffle + order + attribute
        [
            "temp_qa_order_shuffle_phrase_train",
            "temp_qa_order_shuffle_prefix_train",
            "temp_qa_order_shuffle_sentence_train",
            "tempqa_v2_order_long_1x",
            "tempqa_v2_attribute_long_1x",
        ],
        # # shuffle + order + attribute + refer
        [
            "temp_qa_order_shuffle_phrase_train",
            "temp_qa_order_shuffle_prefix_train",
            "temp_qa_order_shuffle_sentence_train",
            "tempqa_v2_order_long_1x",
            "tempqa_v2_attribute_long_1x",
            "temp_qa_refer_anchor_temp2any_train",
            "temp_qa_refer_begin_end_temp2any_train",
        ],
        ["llava_next"],
        # ["llava_next", "hotpotqa_8kLength"],
    ]
    train_datasets = []
    text_dataset_limit = 100_000  # len(tempqa_data) + len(hotpot_qa_llava_format)

    # shuffle data
    for llava_next_ratio in [2]:
        # actual_num
        for total_limit in [
            100_000,
            200_000,
        ]:  # [50_000, 100_000, 200_000, 500_000, 1_000_000]:
            for i, data_composition in enumerate(data_compositions):
                print("Data composition: ", data_composition)
                d = []
                if len(data_composition) == 1:
                    d = data[data_composition[0]][:total_limit]
                else:
                    for k in data_composition:
                        if k != "llava_next":
                            d.extend(data[k][:text_dataset_limit])
                        else:
                            d.extend(
                                data[k][
                                    : (len(data_composition) - 1)
                                    * text_dataset_limit
                                    * llava_next_ratio
                                ]
                            )

                print(data_composition, len(d))
                random.shuffle(d)
                d = d[:total_limit]
                comp_name = (
                    "-".join(data_composition)
                    .replace("tempqa_v2_", "")
                    .replace("temp_qa_", "")
                    .replace("_train", "")
                )  # "refer-shuffle-order-attribute-1-8x"
                # train_datasets.append( [d, comp_name] )
                actual_num = len(d)
                file_name = f"{save_dir}/{comp_name}_per{text_dataset_limit}_total{actual_num}_lRatio{llava_next_ratio}.json"
                if not os.path.exists(file_name):
                    # for d, comp_name in train_datasets:
                    with open(
                        f"{save_dir}/{comp_name}_per{text_dataset_limit}_total{actual_num}_lRatio{llava_next_ratio}.json",
                        "w",
                    ) as f:
                        json.dump(d, f, indent=4)
