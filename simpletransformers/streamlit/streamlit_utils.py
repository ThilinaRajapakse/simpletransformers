import numpy as np

import streamlit as st
import collections
import functools
import inspect
import textwrap
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
import random

def cache_on_button_press(label, **cache_kwargs):
    """Function decorator to memoize function executions.
    Parameters
    ----------
    label : str
        The label for the button to display prior to running the cached funnction.
    cache_kwargs : Dict[Any, Any]
        Additional parameters (such as show_spinner) to pass into the underlying @st.cache decorator.
    Example
    -------
    This show how you could write a username/password tester:
    >>> @cache_on_button_press('Authenticate')
    ... def authenticate(username, password):
    ...     return username == "buddha" and password == "s4msara"
    ...
    ... username = st.text_input('username')
    ... password = st.text_input('password')
    ...
    ... if authenticate(username, password):
    ...     st.success('Logged in.')
    ... else:
    ...     st.error('Incorrect username or password')
    """
    internal_cache_kwargs = dict(cache_kwargs)
    internal_cache_kwargs["allow_output_mutation"] = True
    internal_cache_kwargs["show_spinner"] = False

    def function_decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            @st.cache(**internal_cache_kwargs)
            def get_cache_entry(func, args, kwargs):
                class ButtonCacheEntry:
                    def __init__(self):
                        self.evaluated = False
                        self.return_value = None

                    def evaluate(self):
                        self.evaluated = True
                        self.return_value = func(*args, **kwargs)

                return ButtonCacheEntry()

            cache_entry = get_cache_entry(func, args, kwargs)
            if not cache_entry.evaluated:
                if st.sidebar.button(label):
                    cache_entry.evaluate()
                else:
                    raise st.ScriptRunner.StopException
            return cache_entry.return_value

        return wrapped_func

    return function_decorator


"""Hack to add per-session state to Streamlit.
Usage
-----
>>> import SessionState
>>>
>>> session_state = SessionState.get(user_name='', favorite_color='black')
>>> session_state.user_name
''
>>> session_state.user_name = 'Mary'
>>> session_state.favorite_color
'black'
Since you set user_name above, next time your script runs this will be the
result:
>>> session_state = get(user_name='', favorite_color='black')
>>> session_state.user_name
'Mary'
"""


class SessionState(object):
    def __init__(self, **kwargs):
        """A new SessionState object.
        Parameters
        ----------
        **kwargs : any
            Default values for the session state.
        Example
        -------
        >>> session_state = SessionState(user_name='', favorite_color='black')
        >>> session_state.user_name = 'Mary'
        ''
        >>> session_state.favorite_color
        'black'
        """
        for key, val in kwargs.items():
            setattr(self, key, val)


def get(**kwargs):
    """Gets a SessionState object for the current session.
    Creates a new object if necessary.
    Parameters
    ----------
    **kwargs : any
        Default values you want to add to the session state, if we're creating a
        new one.
    Example
    -------
    >>> session_state = get(user_name='', favorite_color='black')
    >>> session_state.user_name
    ''
    >>> session_state.user_name = 'Mary'
    >>> session_state.favorite_color
    'black'
    Since you set user_name above, next time your script runs this will be the
    result:
    >>> session_state = get(user_name='', favorite_color='black')
    >>> session_state.user_name
    'Mary'
    """
    # Hack to get the session object from Streamlit.

    ctx = get_report_ctx()

    this_session = None

    session_info = Server.get_current()._get_session_info(ctx.session_id)
    this_session = session_info.session

    if this_session is None:
        raise RuntimeError(
            "Oh noes. Couldn't get your Streamlit Session object"
            "Are you doing something fancy with threads?"
        )

    # Got the session object! Now let's attach some state into it.

    if not hasattr(this_session, "_custom_session_state"):
        this_session._custom_session_state = SessionState(**kwargs)

    return this_session._custom_session_state


def simple_transformers_model(model):
    return (type(model).__name__, model.args)


def get_color(i):
    # Colors taken from Sasha Trubetskoy's list of colors - https://sashamaps.net/docs/tools/20-colors/
    colors = [
        (60, 180, 75, 0.4),
        (255, 225, 25, 0.4),
        (0, 130, 200, 0.4),
        (245, 130, 48, 0.4),
        (145, 30, 180, 0.4),
        (70, 240, 240, 0.4),
        (240, 50, 230, 0.4),
        (210, 245, 60, 0.4),
        (250, 190, 212, 0.4),
        (0, 128, 128, 0.4),
        (220, 190, 255, 0.4),
        (170, 110, 40, 0.4),
        (255, 250, 200, 0.4),
        (128, 0, 0, 0.4),
        (170, 255, 195, 0.4),
        (128, 128, 0, 0.4),
        (255, 215, 180, 0.4),
        (0, 0, 128, 0.4),
        (128, 128, 128, 0.4),
        (255, 255, 255, 0.4),
        (0, 0, 0, 0.4),
        (230, 25, 75, 0.4),
    ]

    alt_colors = [
        # maroon	#800000
        (128, 0, 0, 0.4),
        # dark red	#8B0000
        (139, 0, 0, 0.4),
        # brown	#A52A2A
        (165, 42, 42, 0.4),
        # firebrick	#B22222
        (178, 34, 34, 0.4),
        # crimson	#DC143C
        (220, 20, 60, 0.4),
        # red	#FF0000
        (255, 0, 0, 0.4),
        # tomato	#FF6347
        (255, 99, 71, 0.4),
        # coral	#FF7F50
        (255, 127, 80, 0.4),
        # indian red	#CD5C5C
        (205, 92, 92, 0.4),
        # light coral	#F08080
        (240, 128, 128, 0.4),
        # dark salmon	#E9967A
        (233, 150, 122, 0.4),
        # salmon	#FA8072
        (250, 128, 114, 0.4),
        # light salmon	#FFA07A
        (255, 160, 122, 0.4),
        # orange red	#FF4500
        (255, 69, 0, 0.4),
        # dark orange	#FF8C00
        (255, 140, 0, 0.4),
        # orange	#FFA500
        (255, 165, 0, 0.4),
        # gold	#FFD700
        (255, 215, 0, 0.4),
        # dark golden rod	#B8860B
        (184, 134, 11, 0.4),
        # golden rod	#DAA520
        (218, 165, 32, 0.4),
        # pale golden rod	#EEE8AA
        (238, 232, 170, 0.4),
        # dark khaki	#BDB76B
        (189, 183, 107, 0.4),
        # khaki	#F0E68C
        (240, 230, 140, 0.4),
        # olive	#808000
        (128, 128, 0, 0.4),
        # yellow	#FFFF00
        (255, 255, 0, 0.4),
        # yellow green	#9ACD32
        (154, 205, 50, 0.4),
        # dark olive green	#556B2F
        (85, 107, 47, 0.4),
        # olive drab	#6B8E23
        (107, 142, 35, 0.4),
        # lawn green	#7CFC00
        (124, 252, 0, 0.4),
        # chart reuse	#7FFF00
        (127, 255, 0, 0.4),
        # green yellow	#ADFF2F
        (173, 255, 47, 0.4),
        # dark green	#006400
        (0, 100, 0, 0.4),
        # green	#008000
        (0, 128, 0, 0.4),
        # forest green	#228B22
        (34, 139, 34, 0.4),
        # lime	#00FF00
        (0, 255, 0, 0.4),
        # lime green	#32CD32
        (50, 205, 50, 0.4),
        # light green	#90EE90
        (144, 238, 144, 0.4),
        # pale green	#98FB98
        (152, 251, 152, 0.4),
        # dark sea green	#8FBC8F
        (143, 188, 143, 0.4),
        # medium spring green	#00FA9A
        (0, 250, 154, 0.4),
        # spring green	#00FF7F
        (0, 255, 127, 0.4),
        # sea green	#2E8B57
        (46, 139, 87, 0.4),
        # medium aqua marine	#66CDAA
        (102, 205, 170, 0.4),
        # medium sea green	#3CB371
        (60, 179, 113, 0.4),
        # light sea green	#20B2AA
        (32, 178, 170, 0.4),
        # dark slate gray	#2F4F4F
        (47, 79, 79, 0.4),
        # teal	#008080
        (0, 128, 128, 0.4),
        # dark cyan	#008B8B
        (0, 139, 139, 0.4),
        # aqua	#00FFFF
        (0, 255, 255, 0.4),
        # cyan	#00FFFF
        (0, 255, 255, 0.4),
        # light cyan	#E0FFFF
        (224, 255, 255, 0.4),
        # dark turquoise	#00CED1
        (0, 206, 209, 0.4),
        # turquoise	#40E0D0
        (64, 224, 208, 0.4),
        # medium turquoise	#48D1CC
        (72, 209, 204, 0.4),
        # pale turquoise	#AFEEEE
        (175, 238, 238, 0.4),
        # aqua marine	#7FFFD4
        (127, 255, 212, 0.4),
        # powder blue	#B0E0E6
        (176, 224, 230, 0.4),
        # cadet blue	#5F9EA0
        (95, 158, 160, 0.4),
        # steel blue	#4682B4
        (70, 130, 180, 0.4),
        # corn flower blue	#6495ED
        (100, 149, 237, 0.4),
        # deep sky blue	#00BFFF
        (0, 191, 255, 0.4),
        # dodger blue	#1E90FF
        (30, 144, 255, 0.4),
        # light blue	#ADD8E6
        (173, 216, 230, 0.4),
        # sky blue	#87CEEB
        (135, 206, 235, 0.4),
        # light sky blue	#87CEFA
        (135, 206, 250, 0.4),
        # midnight blue	#191970
        (25, 25, 112, 0.4),
        # navy	#000080
        (0, 0, 128, 0.4),
        # dark blue	#00008B
        (0, 0, 139, 0.4),
        # medium blue	#0000CD
        (0, 0, 205, 0.4),
        # blue	#0000FF
        (0, 0, 255, 0.4),
        # royal blue	#4169E1
        (65, 105, 225, 0.4),
        # blue violet	#8A2BE2
        (138, 43, 226, 0.4),
        # indigo	#4B0082
        (75, 0, 130, 0.4),
        # dark slate blue	#483D8B
        (72, 61, 139, 0.4),
        # slate blue	#6A5ACD
        (106, 90, 205, 0.4),
        # medium slate blue	#7B68EE
        (123, 104, 238, 0.4),
        # medium purple	#9370DB
        (147, 112, 219, 0.4),
        # dark magenta	#8B008B
        (139, 0, 139, 0.4),
        # dark violet	#9400D3
        (148, 0, 211, 0.4),
        # dark orchid	#9932CC
        (153, 50, 204, 0.4),
        # medium orchid	#BA55D3
        (186, 85, 211, 0.4),
        # purple	#800080
        (128, 0, 128, 0.4),
        # thistle	#D8BFD8
        (216, 191, 216, 0.4),
        # plum	#DDA0DD
        (221, 160, 221, 0.4),
        # violet	#EE82EE
        (238, 130, 238, 0.4),
        # magenta / fuchsia	#FF00FF
        (255, 0, 255, 0.4),
        # orchid	#DA70D6
        (218, 112, 214, 0.4),
        # medium violet red	#C71585
        (199, 21, 133, 0.4),
        # pale violet red	#DB7093
        (219, 112, 147, 0.4),
        # deep pink	#FF1493
        (255, 20, 147, 0.4),
        # hot pink	#FF69B4
        (255, 105, 180, 0.4),
        # light pink	#FFB6C1
        (255, 182, 193, 0.4),
        # pink	#FFC0CB
        (255, 192, 203, 0.4),
        # antique white	#FAEBD7
        (250, 235, 215, 0.4),
        # beige	#F5F5DC
        (245, 245, 220, 0.4),
        # bisque	#FFE4C4
        (255, 228, 196, 0.4),
        # blanched almond	#FFEBCD
        (255, 235, 205, 0.4),
        # wheat	#F5DEB3
        (245, 222, 179, 0.4),
        # corn silk	#FFF8DC
        (255, 248, 220, 0.4),
        # lemon chiffon	#FFFACD
        (255, 250, 205, 0.4),
        # light golden rod yellow	#FAFAD2
        (250, 250, 210, 0.4),
        # light yellow	#FFFFE0
        (255, 255, 224, 0.4),
        # saddle brown	#8B4513
        (139, 69, 19, 0.4),
        # sienna	#A0522D
        (160, 82, 45, 0.4),
        # chocolate	#D2691E
        (210, 105, 30, 0.4),
        # peru	#CD853F
        (205, 133, 63, 0.4),
        # sandy brown	#F4A460
        (244, 164, 96, 0.4),
        # burly wood	#DEB887
        (222, 184, 135, 0.4),
        # tan	#D2B48C
        (210, 180, 140, 0.4),
        # rosy brown	#BC8F8F
        (188, 143, 143, 0.4),
        # moccasin	#FFE4B5
        (255, 228, 181, 0.4),
        # navajo white	#FFDEAD
        (255, 222, 173, 0.4),
        # peach puff	#FFDAB9
        (255, 218, 185, 0.4),
        # misty rose	#FFE4E1
        (255, 228, 225, 0.4),
        # lavender blush	#FFF0F5
        (255, 240, 245, 0.4),
        # linen	#FAF0E6
        (250, 240, 230, 0.4),
        # old lace	#FDF5E6
        (253, 245, 230, 0.4),
        # papaya whip	#FFEFD5
        (255, 239, 213, 0.4),
        # sea shell	#FFF5EE
        (255, 245, 238, 0.4),
        # mint cream	#F5FFFA
        (245, 255, 250, 0.4),
        # slate gray	#708090
        (112, 128, 144, 0.4),
        # light slate gray	#778899
        (119, 136, 153, 0.4),
        # light steel blue	#B0C4DE
        (176, 196, 222, 0.4),
        # lavender	#E6E6FA
        (230, 230, 250, 0.4),
        # floral white	#FFFAF0
        (255, 250, 240, 0.4),
        # alice blue	#F0F8FF
        (240, 248, 255, 0.4),
        # ghost white	#F8F8FF
        (248, 248, 255, 0.4),
        # honeydew	#F0FFF0
        (240, 255, 240, 0.4),
        # ivory	#FFFFF0
        (255, 255, 240, 0.4),
        # azure	#F0FFFF
        (240, 255, 255, 0.4),
        # snow	#FFFAFA
        (255, 250, 250, 0.4),
        # black	#000000
        (0, 0, 0, 0.4),
        # dim gray / dim grey	#696969
        (105, 105, 105, 0.4),
        # gray / grey	#808080
        (128, 128, 128, 0.4),
        # dark gray / dark grey	#A9A9A9
        (169, 169, 169, 0.4),
        # silver	#C0C0C0
        (192, 192, 192, 0.4),
        # light gray / light grey	#D3D3D3
        (211, 211, 211, 0.4),
        # gainsboro	#DCDCDC
        (220, 220, 220, 0.4),
        # white smoke	#F5F5F5
        (245, 245, 245, 0.4),
        # white	#FFFFFF
        (255, 255, 255, 0.4),
    ]
    try:
        return str(colors[i])
    except IndexError:
        return str(alt_colors[random.randint(a=0, b=len(alt_colors) - 1)])
