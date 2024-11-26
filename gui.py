# main.py

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout import Layout
from prompt_toolkit.widgets import TextArea
from prompt_toolkit.layout.containers import VSplit, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

class MainGUI:
    def __init__(self):
        self.output_text = ""
        self.current_view = None
        self.previous_view = None
        self.setup_ui()

    def setup_ui(self):
        self.kb = KeyBindings()
        self.style = Style.from_dict({
            'output-field': 'bg:#1e1e1e #dcdcdc',
            'input-field': 'bg:#2b2b2b #ffffff',
            'menu': 'bg:#3c3c3c #dcdcdc',
            'line': '#888888',
            'prompt': 'bold #00ff00',
        })

        @self.kb.add('h')
        def _(event):
            self.handle_h_key()

        @self.kb.add('b')
        def _(event):
            self.handle_b_key()

        self.input_field = TextArea(
            height=1,
            prompt='> ',
            style='class:input-field',
            multiline=False,
            wrap_lines=False,
            accept_handler=self.on_enter
        )

        self.output_buffer = Buffer()
        self.output_window = Window(
            content=BufferControl(buffer=self.output_buffer),
            style='class:output-field',
            wrap_lines=True,
            height=Dimension(weight=1)
        )

        self.menu_buffer = Buffer()
        self.menu_control = BufferControl(buffer=self.menu_buffer)

        root_container = VSplit([
            Window(content=self.menu_control, width=40, style='class:menu'),
            HSplit([
                self.output_window,
                Window(height=1, char='─', style='class:line'),
                self.input_field,
            ])
        ])

        self.layout = Layout(root_container, focused_element=self.input_field)
        self.application = Application(
            layout=self.layout,
            key_bindings=self.kb,
            style=self.style,
            full_screen=True,
        )

    def run(self):
        from modules.wave_packet_tunneling.main import WavePacketMenu
        self.current_view = WavePacketMenu(self)
        self.update_display()
        self.application.run()

    def update_display(self):
        self.menu_buffer.text = self.current_view.get_menu_text()
        self.output_buffer.text = self.current_view.get_output()
        self.application.layout.focus(self.input_field)
        self.application.invalidate()

    def on_enter(self, buffer):
        command = buffer.text.strip()
        buffer.text = ''
        if command == 'h':
            self.handle_h_key()
        elif command == 'b':
            self.handle_b_key()