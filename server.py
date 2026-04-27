import litserve as ls
import sys

#from gemini import GeminiPro, GeminiProCfg
from openrouter import OpenRouterAgent as GeminiPro, OpenRouterCfg as GeminiProCfg


timestep = 0

mode = sys.argv[1]

if mode == "inf_base":
    #cfg = GeminiProCfg(max_thinking_tokens=3072, temperature=0.5, mode="inf_base")
    cfg = GeminiProCfg(temperature=0.5, mode="inf_base")
elif mode == "inf_super":
    #cfg = GeminiProCfg(max_thinking_tokens=3072, temperature=0.5, mode="inf_super")
    cfg = GeminiProCfg(temperature=0.5, mode="inf_super")


class SariSariInferenceAPI(ls.LitAPI):
    def setup(self, device):
        self.model = GeminiPro(cfg)

    def decode_request(self, request):
        return request

    def predict(self, request):
        global timestep
        timestep += 1

        return self.model.generate(request, timestep)

    def encode_response(self, response):
        return {'response': response}


if __name__ == "__main__":
    api = SariSariInferenceAPI()
    server = ls.LitServer(api)
    server.run(port=8005)