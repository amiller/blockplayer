import struct

import zmq

# The maximum number of commands in the queue before it gets sent
MAX_COMMANDS = 128

if 'socket' in globals():
    socket.close()

context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect('tcp://*:8134')

cmd_queue = ''
commands = 0

def build_block(x, y, z, typ):
    return 'b' + struct.pack(">hhhh", x, y, z, typ)

def build_message(message):
    return 'p' + message + '\0'

def build_wool(x, y, z, dye):
    return 'w' + struct.pack('>hhhB', x, y, z, dye)

def build_addbounds(x, y, z, length, width, height):
    return 's' + struct.pack('>hhhhhh', x, y, z, length, width, height)

def build_clearbounds():
    return 'c'

def send_queue(queue, num_commands):
    socket.send(struct.pack('>i', num_commands) + queue)

def send_globalqueue():
    global cmd_queue, commands
    send_queue(cmd_queue, commands)
    
    cmd_queue = ''
    commands = 0

def add_queue(command):
    global cmd_queue, commands
    cmd_queue += command
    commands += 1
    
    if commands >= MAX_COMMANDS:
        send_globalqueue()

class DyeColor:
    WHITE = 0x0
    ORANGE = 0x1
    MAGENTA = 0x2
    LIGHT_BLUE = 0x3
    YELLOW = 0x4
    LIME = 0x5
    PINK = 0x6
    GRAY = 0x7
    SILVER = 0x8
    CYAN = 0x9
    PURPLE = 0xA
    BLUE = 0xB
    BROWN = 0xC
    GREEN = 0xD
    RED = 0xE
    BLACK = 0xF


class Material:
    AIR = 0
    GRASS = 2
    GLASS = 20
    FENCE = 85


class Board:
    BOARD_LENGTH = 32;
    BOARD_WIDTH = 12;
    BOARD_HEIGHT = 9;
    
    def __init__(self):
        self.wall_offset = 0
    
    def set_block(self, x, y, z, typ):
        add_queue(build_block(x, y, z, typ))
    
    def set_wool(self, x, y, z, dye):
        add_queue(build_wool(x, y, z, dye))
    
    def add_bound(self, x, y, z, length, width, height):
        add_queue(build_addbounds(x, y, z, length, width, height))
    
    def clear_bounds(self):
        add_queue(build_clearbounds())
    
    def setup_gameboard(self):
        """Builds the game board and sends it to the server"""
        
        # Clear the board
        for x in xrange(self.BOARD_LENGTH+1):
            for z in xrange(self.BOARD_WIDTH+1):
                for y in xrange(self.BOARD_HEIGHT+1):
                    self.set_block(x, y, z, Material.AIR)
                    self.set_block(z, y, x, Material.AIR)
        
        # First, the perimeter of grass
        for x in xrange(self.BOARD_LENGTH+1):
            self.set_block(x, -1, 0, Material.GRASS)
            self.set_block(0, -1, x, Material.GRASS)
        for x in xrange(self.BOARD_LENGTH-self.BOARD_WIDTH+1):
            self.set_block(self.BOARD_WIDTH+x, -1, self.BOARD_WIDTH, Material.GRASS)
            self.set_block(self.BOARD_WIDTH, -1, self.BOARD_WIDTH+x, Material.GRASS)
        
        # Next, the purple border
        for x in xrange(self.BOARD_LENGTH):
            self.set_wool(x+1, -1, 1, DyeColor.PURPLE)
            self.set_wool(1, -1, x+1, DyeColor.PURPLE)
        for x in xrange(self.BOARD_LENGTH-self.BOARD_WIDTH+2):
            self.set_wool(self.BOARD_WIDTH+x-1, -1, self.BOARD_WIDTH-1, DyeColor.PURPLE)
            self.set_wool(self.BOARD_WIDTH-1, -1, self.BOARD_WIDTH+x-1, DyeColor.PURPLE)
        
        # The fence border
        for x in xrange(self.BOARD_LENGTH+1):
            self.set_block(x, 0, 0, Material.FENCE)
            self.set_block(0, 0, x, Material.FENCE)
        for x in xrange(self.BOARD_LENGTH-self.BOARD_WIDTH+1):
            self.set_block(self.BOARD_WIDTH+x, 0, self.BOARD_WIDTH, Material.FENCE)
            self.set_block(self.BOARD_WIDTH, 0, self.BOARD_WIDTH+x, Material.FENCE)
        
        # Clear the origin
        self.set_block(0, 0, 0, Material.AIR)
        # Add a block for the player to jump on to get over the fences
        self.set_wool(1, 0, 1, DyeColor.PURPLE)
        
        # The black playing board
        for x in xrange(2, self.BOARD_LENGTH+1):
            # The grass and purple border add remove 2 from each side
            for z in xrange(2, self.BOARD_WIDTH-2+1):
                self.set_wool(x, -1, z, DyeColor.BLACK)
                self.set_wool(z, -1, x, DyeColor.BLACK)
        
        # The blue playing area marker
        for x in xrange(2, self.BOARD_WIDTH-2+1):
            self.set_wool(x, -1, self.BOARD_WIDTH-1, DyeColor.BLUE)
            self.set_wool(self.BOARD_WIDTH-1, -1, x, DyeColor.BLUE)
        
        # The glass ceiling
        for x in xrange(self.BOARD_LENGTH+1):
            for z in xrange(self.BOARD_WIDTH+1):
                self.set_block(x, self.BOARD_HEIGHT+1, z, Material.GLASS)
                self.set_block(z, self.BOARD_HEIGHT+1, x, Material.GLASS)
        
        # Draw the blank game walls
        self.draw_gamewalls(0, True)
        
        # Set the bounds
        self.add_bound(-1, -1, -1, self.BOARD_LENGTH+2, self.BOARD_WIDTH+2,
            self.BOARD_HEIGHT+1)
        self.add_bound(-1, -1, -1, self.BOARD_WIDTH+2, self.BOARD_LENGTH+2,
            self.BOARD_HEIGHT+1)
        
        # Flush the command queue
        send_globalqueue()
    
    def draw_gamewalls(self, offset=0, blank=False):
        """Draws the walls where the designs appear at an offset from the end of
        the board. If blank is True, the designs are not drawn."""
        
        # Clear the old walls
        if offset != self.wall_offset:
            for x in xrange(1, self.BOARD_WIDTH-1+1):
                for y in xrange(self.BOARD_HEIGHT+1):
                    # x-side (left from origin) wall
                    self.set_block(self.BOARD_LENGTH-self.wall_offset, y, x, Material.AIR)
                    # z-side (right from origin) wall
                    self.set_block(x, y, self.BOARD_LENGTH-self.wall_offset, Material.AIR)
        
        # Don't intersect the walls if they're in the playing area
        intersect = 0
        if offset > self.BOARD_LENGTH-self.BOARD_WIDTH:
            intersect = offset-(self.BOARD_LENGTH-self.BOARD_WIDTH)-1
        
        for x in xrange(1, self.BOARD_WIDTH-1+1-intersect):
            for y in xrange(self.BOARD_HEIGHT+1):
                # x-side (left from origin) wall
                self.set_wool(self.BOARD_LENGTH-offset, y, x, DyeColor.YELLOW)
                # z-side (right from origin) wall
                self.set_wool(x, y, self.BOARD_LENGTH-offset, DyeColor.YELLOW)
        
        self.wall_offset = offset
        send_globalqueue()


class Game:
    def __init__(self, clear_bounds=True):
        self.board = Board()
        
        if clear_bounds:
            self.board.clear_bounds()
        
        self.board.setup_gameboard()


if __name__ == '__main__':
    game = Game()

