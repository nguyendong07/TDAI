from django.shortcuts import render

# Create your views here.
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import Extract
from .serializer import ExtractImage
from .module import ImageModule

class ImageView(APIView):
    def get(self, request):
        image = Extract.objects.all()
        serializer = ExtractImage(data=image, many=True)
        if serializer.is_valid():
            print('serializer.data')
        print(serializer.data)
        return Response({'message': 'ok', 'data': serializer.data}, status=status.HTTP_200_OK)

    def post(self, request):
        serializer = ExtractImage(data=request.data)
        if serializer.is_valid():
            serializer.save()
            path = serializer.data['content']
            img = ImageModule(path)
            if not img['success']:
                pass
        return Response(img, status=status.HTTP_200_OK)