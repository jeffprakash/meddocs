import openai
from rest_framework.views import APIView
from rest_framework.response import Response

class ChatGPTView(APIView):
    def post(self, request):
        input_text = request.data.get('input_text', '')
        openai.api_key = "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"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=input_text,
            max_tokens=200,
            temperature=0.2,

        )
        return Response({'response': response.choices[0].text})