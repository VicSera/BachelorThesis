from core.config import Config


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


def normalize(value, max, min=0):
    return (value - min) / (max - min)


def denormalize(value, max, min=0):
    return value * (max - min) + min


def clamp(value, maximum=Config.MAX_VELOCITY - 1, minimum=0):
    return max(minimum, min(value, maximum))

