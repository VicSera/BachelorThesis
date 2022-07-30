def format_input_output(osWalkEntry):
    try:
        return {
            'directory': osWalkEntry[0],
            'bass': f'{osWalkEntry[0]}/{osWalkEntry[2][0]}',
            'drums': f'{osWalkEntry[0]}/{osWalkEntry[2][1]}',
            'other': f'{osWalkEntry[0]}/{osWalkEntry[2][2]}',
            'vocals': f'{osWalkEntry[0]}/{osWalkEntry[2][3]}',
        }
    except:
        return {}
