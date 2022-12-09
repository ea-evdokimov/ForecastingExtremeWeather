import typing as tp
import numpy as np
from pydantic import BaseModel


class WindDescription(BaseModel):
    isnan: bool
    changed: bool
    x_rad: float
    y_rad: float


def dd_preparation(description: tp.Optional[str]) -> WindDescription:
    '''
    :param description: string description of wind
    :return: WindDescription object
    '''
    if description is None:
        return WindDescription(isnan=True, changed=False, x_rad=0., y_rad=0.)

    prepared_description = description.lower()

    if 'переменное направление'.lower() in prepared_description:
        return WindDescription(isnan=False, changed=True, x_rad=0., y_rad=0.)

    if 'штиль' in prepared_description or 'безветрие' in prepared_description:
        return WindDescription(isnan=False, changed=False, x_rad=0., y_rad=0.)

    dest_to_degree = {'восток': 0.,
                      'север': np.pi / 2,
                      'запад': np.pi,
                      'юг': 3 * np.pi / 2}

    angle = 0
    count = 0
    for dest, angle in dest_to_degree.items():
        angle += angle * description.count(dest)
        count += description.count(dest)
    res_angle = angle / count
    return WindDescription(isnan=False, changed=False, x_rad=np.cos(res_angle), y_rad=np.sin(res_angle))


# ['Ветер, дующий с северо-северо-запада',
#  'Ветер, дующий с северо-запада',
#  'Ветер, дующий с юга',
#  'Ветер, дующий с запада',
#  'Ветер, дующий с западо-северо-запада',
#  'Ветер, дующий с западо-юго-запада',
#  'Ветер, дующий с юго-юго-востока',
#  'Ветер, дующий с юго-юго-запада',
#  'Ветер, дующий с юго-востока',
#  'Ветер, дующий с востоко-юго-востока',
#  'Ветер, дующий с юго-запада',
#  'Штиль, безветрие',
#  'Ветер, дующий с севера',
#  'Ветер, дующий с северо-северо-востока',
#  'Ветер, дующий с востока',
#  'Ветер, дующий с северо-востока',
#  'Ветер, дующий с востоко-северо-востока',
#  'Переменное направление']
