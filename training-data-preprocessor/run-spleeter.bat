docker pull deezer/spleeter:3.8-4stems

docker run -v "C:\programming\BachelorThesis\data\raw":"/input" -v "C:\programming\BachelorThesis\data\split":"/output" -v "C:\programming\BachelorThesis\training-data-preprocessor\split.sh":"/split.sh" --entrypoint="/split.sh" --rm deezer/spleeter:3.8-4stems