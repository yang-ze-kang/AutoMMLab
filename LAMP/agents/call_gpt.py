import openai
from openai import AzureOpenAI
import os
import copy
import pkg_resources
from packaging import version
import random


last_openai_request_cost = 0.0


def get_last_openai_request_cost():
    return last_openai_request_cost


USE_AZURE = True

if USE_AZURE:
    API_VERSION = "2023-12-01-preview"
    if not (
        os.environ.get("AZURE_OPENAI_ENDPOINT")
        and os.environ.get("AZURE_OPENAI_KEY")
        and os.environ.get("AZURE_API_VERSION")
    ):
        model_to_region_and_tpm = {
            "gpt-4-vision-preview": [("us-w", 10), ("se-c", 10)],
            "gpt-4-1106-preview": [
                ("us-w", 100),
                ("in-s", 100),
                # ("no-e", 150),
                # ("se-c", 150),
                # ("au-e", 80),
                # ("ca-e", 80),
                # ("us-e2", 80),
                # ("fr-c", 80),
                # ("uk-s", 80),
            ],
            "gpt-35-turbo-1106": [("us-w", 100), ("in-s", 100)],
        }
        region_to_api_key = {
            "us-w": "",
            "us-e2": "",
            "fr-c": "",
            "uk-s": "",
        }
        # example: {model: [(client1, weight1), (client2, weight2), ...)]}
        model_to_client_and_tpm = {
            model: [
                (
                    AzureOpenAI(
                        azure_endpoint=f"",
                        api_key=region_to_api_key[region],
                        api_version=API_VERSION,
                    ),
                    tpm,
                )
                for region, tpm in model_to_region_and_tpm[model]
            ]
            for model in model_to_region_and_tpm.keys()
        }
    else:
        # TODO
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.environ.get("AZURE_API_VERSION"),
        )
else:
    # TODO 待验证
    if not os.environ.get("OPENAI_API_KEY"):
        openai.api_key = "fk221315-UIp73CUkRiNI7dzlBS9OEdxrTjkgNsaX"
        # openai.api_base = "https://openai.api2d.net/v1"
        try:
            if version.parse(pkg_resources.get_distribution("openai").version) < version.parse("1"):
                openai.api_base = "https://oa.ai01.org/v1"
            else:
                openai.base_url = "https://oa.ai01.org/v1/"
        except pkg_resources.DistributionNotFound:
            print("openai is not installed.")



if not os.environ.get("OPENAI_API_KEY"):
    openai.api_key = "fk221315-UIp73CUkRiNI7dzlBS9OEdxrTjkgNsaX"
    # openai.api_base = "https://openai.api2d.net/v1"
    try:
        if version.parse(pkg_resources.get_distribution("openai").version) < version.parse("1"):
            openai.api_base = "https://oa.ai01.org/v1"
        else:
            openai.base_url = "https://oa.ai01.org/v1/"
    except pkg_resources.DistributionNotFound:
        print("openai is not installed.")



def call_openai_completion_api(messages, model="gpt-4", **kwargs):
    messages_for_log = copy.deepcopy(messages)
    for msg in messages_for_log:
        for content in msg["content"]:
            if isinstance(content, dict) and content.get("image_url"):
                content["image_url"] = "data:image/jpeg;base64, ..."
    # print(messages_for_log)

    if version.parse(pkg_resources.get_distribution("openai").version) < version.parse("1"):
        # 暂不维护openai<1.0
        response = openai.ChatCompletion.create(model=model, messages=messages, **kwargs)
    else:
        if USE_AZURE:
            client = random.choices(
                population=[client for client, _ in model_to_client_and_tpm[model]],
                weights=[tpm for _, tpm in model_to_client_and_tpm[model]],
                k=1,
            )[0]
            response = client.chat.completions.create(model=model, messages=messages, **kwargs)
        else:
            # TODO 待验证
            response = openai.chat.completions.create(model=model, messages=messages, **kwargs)

    if kwargs.get("stream") == True:
        collected_messages = []
        for chunk in response:
            chunk_message = chunk["choices"][0]["delta"]  # extract the message
            collected_messages.append(chunk_message)  # save the message
            # print(chunk_message.get('content', ''))
        full_reply_content = "".join([m.get("content", "") for m in collected_messages])
        # print(full_reply_content)
        return full_reply_content
    # print(response)
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    # gpt4-turbo cost
    global last_openai_request_cost
    last_openai_request_cost = (0.01 * prompt_tokens + 0.03 * completion_tokens) / 1000

    return response.choices[0].message.content

def gpt4(messages, model="gpt-4-1106-preview", **kwargs):
    messages_for_log = copy.deepcopy(messages)
    for msg in messages_for_log:
        for content in msg["content"]:
            if isinstance(content, dict) and content.get("image_url"):
                content["image_url"] = "data:image/jpeg;base64, ..."
    # print(messages_for_log)

    if version.parse(pkg_resources.get_distribution("openai").version) < version.parse("1"):
        # 暂不维护openai<1.0
        response = openai.ChatCompletion.create(model=model, messages=messages, **kwargs)
    else:
        if USE_AZURE:
            client = random.choices(
                population=[client for client, _ in model_to_client_and_tpm[model]],
                weights=[tpm for _, tpm in model_to_client_and_tpm[model]],
                k=1,
            )[0]
            response = client.chat.completions.create(model=model, messages=messages, **kwargs)
        else:
            # TODO 待验证
            response = openai.chat.completions.create(model=model, messages=messages, **kwargs)

    if kwargs.get("stream") == True:
        collected_messages = []
        for chunk in response:
            chunk_message = chunk["choices"][0]["delta"]  # extract the message
            collected_messages.append(chunk_message)  # save the message
            # print(chunk_message.get('content', ''))
        full_reply_content = "".join([m.get("content", "") for m in collected_messages])
        return full_reply_content
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    # gpt4-turbo cost
    global last_openai_request_cost
    last_openai_request_cost = (0.01 * prompt_tokens + 0.03 * completion_tokens) / 1000

    return response

def gpt35(messages, model="gpt-35-turbo-1106", **kwargs):
    messages_for_log = copy.deepcopy(messages)
    for msg in messages_for_log:
        for content in msg["content"]:
            if isinstance(content, dict) and content.get("image_url"):
                content["image_url"] = "data:image/jpeg;base64, ..."
    # print(messages_for_log)

    if version.parse(pkg_resources.get_distribution("openai").version) < version.parse("1"):
        # 暂不维护openai<1.0
        response = openai.ChatCompletion.create(model=model, messages=messages, **kwargs)
    else:
        if USE_AZURE:
            client = random.choices(
                population=[client for client, _ in model_to_client_and_tpm[model]],
                weights=[tpm for _, tpm in model_to_client_and_tpm[model]],
                k=1,
            )[0]
            response = client.chat.completions.create(model=model, messages=messages, **kwargs)
        else:
            # TODO 待验证
            response = openai.chat.completions.create(model=model, messages=messages, **kwargs)

    if kwargs.get("stream") == True:
        collected_messages = []
        for chunk in response:
            chunk_message = chunk["choices"][0]["delta"]  # extract the message
            collected_messages.append(chunk_message)  # save the message
            # print(chunk_message.get('content', ''))
        full_reply_content = "".join([m.get("content", "") for m in collected_messages])
        return full_reply_content
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    # gpt4-turbo cost
    global last_openai_request_cost
    last_openai_request_cost = (0.01 * prompt_tokens + 0.03 * completion_tokens) / 1000

    return response

openai_model_list = [
    "gpt-4",
    "gpt-4-32k",
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-1106",
    "text-embedding-ada-002",
]



if __name__ == "__main__":
    # _test_glm4v()
    # exit()
    messages = [{"role": "system", "content": "test"}, {"role": "user", "content": "你是谁?"}]
    # model = "gpt-3.5-turbo-1106"
    model = "gpt-4-1106-preview"
    kwargs = {
        "temperature": 0.01,
        "stream": False,
        # "max_new_tokens": 10
    }
    result = call_openai_completion_api(messages, model, **kwargs)
    print(result)
