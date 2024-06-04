import json
import time

import eval_utils.api_credentials as api_credentials
import openai


class OPENAIBaseEngine():
    def __init__(self, engine, azure='True'):
        self.engine = engine
        self.credentials = api_credentials.get_credentials(engine, azure)
        print(self.credentials)
        self.MAX_ATTEMPTS = 10
        self.RATE_WAITTIME = 10
        self.ERROR_WAITTIME = 10

    def safe_completion(self,
                             prompt: str,
                             max_tokens: int = 800,
                             temperature: float = 0,
                             top_p: float = 1,
                             **kwargs):
        assert not (temperature > 0.0 and top_p < 1.0)
        args_dict = self.credentials["api_args"]
        args_dict.update(
            {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        if top_p == 1.0: args_dict.pop("top_p")

        last_exc = None
        for i in range(self.MAX_ATTEMPTS):
            try:
                if self.credentials['mode'] == 'Completion':
                    response = openai.Completion.create(prompt=prompt, **args_dict)
                elif self.credentials['mode'] == 'Chat':
                    response = openai.ChatCompletion.create(messages=prompt, **args_dict)
                response = json.loads(json.dumps(response))
                response_out = {
                    "finish_reason": response["choices"][0]["finish_reason"],
                    "content": response["choices"][0]["message"]["content"]
                }
                # self.total_usage += response["usage"]["total_tokens"] #not defined
                return response_out

            # wait longer for rate limit errors
            except openai.error.InvalidRequestError as e:
                response_out = {"finish_reason": "invalid_request", "content": None}
                response_out = {"finish_reason": "api_error", "content": " OPENAI Error:" + str(e)}
                return response_out
            except openai.error.RateLimitError as e:
                last_exc = e
                time.sleep(self.RATE_WAITTIME)
            except openai.error.OpenAIError as e:
                last_exc = e
                time.sleep(self.ERROR_WAITTIME)
            except Exception as e:
                last_exc = e
                time.sleep(self.ERROR_WAITTIME)

        if isinstance(last_exc, openai.error.RateLimitError):
            raise RuntimeError("Consistently hit rate limit error")

        # make placeholder choices
        if self.credentials['mode'] == 'Completion':
            fake_choice = {
                    "content": " OPENAI Error:" + str(last_exc),
                    "finish_reason": "api_error"
                }

        elif self.credentials['mode'] == 'Chat':
            fake_choice = {
                    "content": " OPENAI Error:" + str(last_exc),
                    "finish_reason": "api_error"
                }
        return fake_choice

    def test_api(self):
        prompt = 'Why did the chicken cross the road?'
        if self.credentials['mode'] == 'Chat':
            prompt = [{"role": "user", "content": "Why did the chicken cross the road?"}]
        response = self.safe_completion(max_tokens=20, temperature=0, top_p=1.0, prompt=prompt)

        if response["finish_reason"] == 'api_error':
            if self.credentials['mode'] == 'Chat':
                print(f'Error in connecting to API: {response[0]["message"]["content"]}')
            else:
                print(f'Error in connecting to API: {response[0]["text"]}')
            exit()
        else:
            print(f'Successful API connection')