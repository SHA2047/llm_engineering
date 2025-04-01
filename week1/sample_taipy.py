from taipy import Gui

user_list = ""
summary_display = ""

page = """
<|{user_list}|input|label=Enter Page url|>

<|Breakit|button|on_action=break_list|>

<|{summary_display}|text|>
"""

def break_list(state):
    state.summary_display = state.user_list

gui = Gui(page)
gui.run(port=8080,use_reloader=True)