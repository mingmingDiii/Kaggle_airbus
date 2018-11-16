from main_code_seg.models.se154_normal import SE154_NORMAL
from main_code_seg.models.link_dense import LINK_DENSE
from main_code_seg.models.dpn_normal import DPN_NORMAL

from main_code_seg.models.dense_normal import DENSE_NORMAL
from main_code_seg.models.dense_scse import DENSE_SCSE
from main_code_seg.models.dense_oc import DENSE_OC
from main_code_seg.models.dn_2c import DN_C2
from main_code_seg.models.dn_5c import DN_5C
from main_code_seg.models.dn_4c import DN_4C
from main_code_seg.models.dn_4c_s import DN_4C_S
from main_code_seg.models.dn_8c import DN_8C
models = {



    'se154_normal':lambda :SE154_NORMAL(),
    'link_dense':lambda :LINK_DENSE(),
    'dpn_normal':lambda :DPN_NORMAL(),
    'dense_normal':lambda :DENSE_NORMAL(),
    'dense_scse':lambda :DENSE_SCSE(),
    'dense_oc':lambda :DENSE_OC(),
    'dn_2c':lambda :DN_C2(),
    'dn_5c':lambda :DN_5C(),
    'dn_4c':lambda :DN_4C(),
    'dn_4c_s':lambda :DN_4C_S(),
    'dn_8c':lambda :DN_8C()





}