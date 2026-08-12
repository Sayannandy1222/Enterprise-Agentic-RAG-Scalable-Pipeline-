"""Optional HTTP adapters for Portkey and Groq-compatible chat APIs."""
from __future__ import annotations
import json
from urllib.request import Request, urlopen
from app.gateway.service import LLMRequest, LLMResponse, _estimate_tokens

class OpenAICompatibleProvider:
    def __init__(self, name, api_key, base_url, model, extra_headers=None, timeout=30):
        self.name=name; self.api_key=api_key; self.base_url=base_url.rstrip('/'); self.model=model; self.extra_headers=extra_headers or {}; self.timeout=timeout
    def generate(self, request: LLMRequest) -> LLMResponse:
        messages=[]
        if request.system_prompt: messages.append({"role":"system","content":request.system_prompt})
        messages.append({"role":"user","content":request.prompt})
        payload=json.dumps({"model":request.model or self.model,"messages":messages,"temperature":request.temperature,"max_tokens":request.max_tokens}).encode()
        headers={"Authorization":f"Bearer {self.api_key}","Content-Type":"application/json",**self.extra_headers}
        with urlopen(Request(f"{self.base_url}/chat/completions",data=payload,headers=headers,method="POST"),timeout=self.timeout) as response:
            body=json.loads(response.read())
        text=body["choices"][0]["message"]["content"]
        usage=body.get("usage",{})
        return LLMResponse(text,self.name,body.get("model",self.model),usage.get("prompt_tokens",_estimate_tokens(request.prompt)),usage.get("completion_tokens",_estimate_tokens(text)),usage.get("total_tokens",0),metadata={"raw_usage":usage})

class GroqProvider(OpenAICompatibleProvider):
    def __init__(self, api_key, model="llama-3.3-70b-versatile", **kwargs): super().__init__("groq",api_key,"https://api.groq.com/openai/v1",model,**kwargs)

class PortkeyProvider(OpenAICompatibleProvider):
    def __init__(self, api_key, virtual_key, model="llama-3.3-70b-versatile", **kwargs): super().__init__("portkey",api_key,"https://api.portkey.ai/v1",model,{"x-portkey-virtual-key":virtual_key},**kwargs)
