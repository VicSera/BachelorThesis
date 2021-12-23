def format_input_output(osWalkEntry):
    try:
        return {
            'directory': osWalkEntry[0],
            'input': f'{osWalkEntry[0]}/{osWalkEntry[2][1]}',
            'output': f'{osWalkEntry[0]}/{osWalkEntry[2][0]}'
        }
    except:
        return {}
