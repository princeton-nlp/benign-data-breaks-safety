import os

engine_env_mappings = {
    "gpt-35-turbo-0301": {
        "OPENAI_API_KEY": "OPENAI_API_KEY_3",
        "OPENAI_ORG_ID": "OPENAI_ORG_ID_3",
        "api": "openai"
    },
    "gpt-35-turbo-16k": {
        "OPENAI_API_KEY": "OPENAI_API_KEY_3",
        "OPENAI_ORG_ID": "OPENAI_ORG_ID_3",
        "api": "openai"
    },
    "gpt-4": {
        "OPENAI_API_KEY": "OPENAI_API_KEY_4",
        "OPENAI_ORG_ID": "OPENAI_ORG_ID_4",
        "api": "openai"
    },
    "gpt-4-1106-preview": {
        "OPENAI_API_KEY": "OPENAI_API_KEY_4",
        "OPENAI_ORG_ID": "OPENAI_ORG_ID_4",
        "api": "openai"
    },
}

def get_credentials(engine, azure=None):
    if azure:
        return {
            "api_args": {
                "api_key": os.environ.get(engine_env_mappings[engine]["OPENAI_API_KEY"]),
                "api_base": os.environ.get(engine_env_mappings[engine]["OPENAI_ORG_ID"]),
                "api_type": 'azure',
                "api_version": '2023-05-15',
                "engine": engine,
            },
            "mode": 'Chat'
        }
    else:
        return {
            "api_args": {
                "api_key": os.environ.get(engine_env_mappings[engine]["OPENAI_API_KEY"]),
                "engine": engine,
            },
            "mode": 'Chat'
        }


