import struct
import random
import os.path
from glob import glob
from threading import Timer, Thread

import zmq
import numpy as np

from blockplayer import main
from blockplayer import stencil
from blockplayer import config
from blockplayer import blockcraft
from blockplayer import blockdraw
from blockplayer import dataset
from blockplayer import glxcontext
import opennpy

FOR_REAL = True
# The maximum number of commands in the queue before it gets sent
MAX_COMMANDS = 128

class HitBException(Exception):
    pass


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
    BOARD_LENGTH = 40;
    BOARD_WIDTH = 12;
    BOARD_HEIGHT = 9;
    
    PLAY_WIDTH = 9
    PLAY_HEIGHT = BOARD_HEIGHT
    
    # The tracks on the board that help the player determine where to place
    # their blocks. Length should be PLAY_WIDTH
    TRACKS = [
        DyeColor.ORANGE,
        DyeColor.MAGENTA,
        DyeColor.SILVER,
        DyeColor.CYAN,
        DyeColor.WHITE,
        DyeColor.PINK,
        DyeColor.BROWN,
        DyeColor.LIME,
        DyeColor.GRAY
    ]
    
    def __init__(self):
        self.wall_offset = 0
        
        self.x_design = None
        self.z_design = None
    
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
                self.set_wool(x, -1, z, Board.TRACKS[z-2])
                self.set_wool(z, -1, x, Board.TRACKS[z-2])
        
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
        self.add_bound(0, -1, 0, self.BOARD_LENGTH, self.BOARD_WIDTH,
            self.BOARD_HEIGHT+2)
        self.add_bound(0, -1, 0, self.BOARD_WIDTH, self.BOARD_LENGTH,
            self.BOARD_HEIGHT+2)
        
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
        
        # Setup blank yellow walls
        for x in xrange(1, self.BOARD_WIDTH-1+1):
            for y in xrange(self.BOARD_HEIGHT+1):
                # x-side (left from origin) wall
                self.set_wool(self.BOARD_LENGTH-offset, y, x, DyeColor.YELLOW)
                
                # z-side (right from origin) wall
                self.set_wool(x, y, self.BOARD_LENGTH-offset, DyeColor.YELLOW)
        
        # Clear away any design
        if not blank:
            if self.x_design is not None:
                for x,row in enumerate(self.x_design):
                    for y,isset in enumerate(row):
                        if isset:
                            self.set_block(self.BOARD_LENGTH-offset, y, x+2, Material.AIR)
            
            if self.z_design is not None:
                for x,row in enumerate(self.z_design):
                    for y,isset in enumerate(row):
                        if isset:
                            self.set_block(x+2, y, self.BOARD_LENGTH-offset, Material.AIR)
        
        self.wall_offset = offset
        send_globalqueue()
    
    def load_design(self, path):
        """Load in the x and z wall designs from the file at `path`"""
        with open(path, 'r') as fp:
            lines = fp.readlines(self.PLAY_WIDTH)
            if len(lines) != self.PLAY_HEIGHT*2 + 1:
                raise HitBException("Cutout design file \"%s\" incorrectly formatted." % path)
            
            # Create the empty design lists
            self.x_design = []
            self.z_design = []
            for x in xrange(self.PLAY_WIDTH):
                self.x_design.append([False]*self.PLAY_HEIGHT)
                self.z_design.append([False]*self.PLAY_HEIGHT)
            
            # Fill the x-side design
            for y,line in enumerate(lines[:self.PLAY_HEIGHT]):
                for x,c in enumerate(line):
                    if c == '\n': continue
                    self.x_design[x][self.PLAY_HEIGHT-y-1] = (c == '#')
            
            # Fill the z-side design
            for y,line in enumerate(lines[self.PLAY_HEIGHT+1:]):
                for x,c in enumerate(line):
                    if c == '\n': continue
                    self.z_design[x][self.PLAY_HEIGHT-y-1] = (c == '#')
    
    def clear_design(self):
        """Clears both wall designs"""
        self.x_design = []
        self.z_design = []
    
    def update_blocks(self, blocks, remove=True):
        """Draw the blocks in the playing area. The blocks array must be of size
        PLAY_WIDTH x PLAY_WIDTH x PLAY_HEIGHT. If remove is True, it clears
        any block with a False value in `blocks`"""
        if blocks is None:
            raise HitBException("Invalid blocks array passed to update function. Cannot update blocks.")
        
        if blocks.shape != (Board.PLAY_WIDTH, Board.PLAY_HEIGHT, Board.PLAY_WIDTH):
            raise HitBException("Blocks array has wrong dimensions. Cannot update blocks.")
        
        for x in xrange(Board.PLAY_WIDTH):
            for y in xrange(Board.PLAY_HEIGHT):
                for z in xrange(Board.PLAY_WIDTH):
                    if blocks[x,y,z]:
                        self.set_wool(x+2, y, z+2, DyeColor.LIGHT_BLUE)
                    elif remove:
                        self.set_block(x+2, y, z+2, Material.AIR)
        
        send_globalqueue()


class Game:
    STATE_QUIT = -1
    STATE_NOTPLAYING = 0
    STATE_LOADDESIGN = 1
    STATE_COUNTDOWN = 2
    STATE_MOVEWALL = 3
    STATE_CHECKWALL = 4
    
    def __init__(self, clear_bounds=True, block_thread=None):
        self.state = Game.STATE_NOTPLAYING
        self.countdown = 5
        self.round = 0
        self.score = 0
        self.wall_offset = 0
        
        self.block_loop_quit = False
        self.blocks = None
        if block_thread is None:
            self.block_thread = Thread(target=self.block_setup)
        else:
            self.block_thread = block_thread
        
        self.board = Board()
        
        if clear_bounds:
            self.board.clear_bounds()
        
        self.board.setup_gameboard()
        
        # Find all the designs in the cutouts directory
        self.designs = []
        self.used_designs = []
        self.designs_dir = os.path.join('.', os.path.dirname(__file__), 'cutouts')
        if os.path.exists(self.designs_dir):
            self.designs = glob(self.designs_dir + '/*')
            random.shuffle(self.designs)
        
        # Initialize blockplayer
        self.block_thread.start()
    
    def quit(self):
        """Stops game processing"""
        self.state = Game.STATE_QUIT
        self.block_loop_quit = True
    
    def send_message(self, message):
        add_queue(build_message(message))
        send_globalqueue()
    
    def load_design(self):
        """Loads the next design and pops it into used_designs"""
        if len(self.designs) == 0:
            if len(self.used_designs) == 0:
                print "No designs available!"
                return
            self.designs = self.used_designs
            random.shuffle(self.designs)
            self.used_designs = []
        
        design = self.designs.pop()
        self.board.load_design(design)
        self.used_designs.append(design)
    
    def block_setup(self):
        """Initialize blockplayer stuff"""
        glxcontext.init()
        print "GL Version String: ",
        glxcontext.printinfo()
        
        main.initialize()
        
        config.load('data/newest_calibration')
        opennpy.align_depth_to_rgb()
        dataset.setup_opencl()
        
        self.block_loop_quit = False
        self.block_loop()
    
    def block_slice(self, blocks):
        """Slices `blocks` to fit the playing area."""
        l,h,w = blocks.shape
        return (blocks[l/2-int(ceil(float(Board.PLAY_WIDTH)/2)):l/2+Board.PLAY_WIDTH/2,
            :Board.PLAY_HEIGHT,w/2-int(ceil(float(Board.PLAY_WIDTH)/2)):w/2+Board.PLAY_WIDTH/2])
    
    def block_once(self):
        """Process the next blockplayer frame"""
        opennpy.sync_update()
        depth,_ = opennpy.sync_get_depth()
        rgb,_ = opennpy.sync_get_video()
        
        main.update_frame(depth, rgb)
        
        # Update the blocks in the playing area
        if self.state != Game.STATE_CHECKWALL:
            self.blocks = blockcraft.translated_rotated(main.R_correct, main.grid.occ)
            self.blocks = self.block_slice(self.blocks)
            self.board.update_blocks(self.blocks)
    
    def block_loop(self):
        """Loops the blockplayer frame processing function"""
        while not self.block_loop_quit:
            self.block_once()
    
    def start(self):
        """Begins the game."""
        if self.state == Game.STATE_QUIT:
            return
        
        if self.state == Game.STATE_NOTPLAYING:
            self.countdown = 5
            self.round += 1
            self.wall_offset = 0
            self.board.clear_design()
            
            self.state = Game.STATE_LOADDESIGN
        
        if self.state == Game.STATE_LOADDESIGN:
            self.load_design()
            self.board.draw_gamewalls()
            self.state = Game.STATE_COUNTDOWN
        
        if self.state == Game.STATE_COUNTDOWN:
            if self.countdown == 0:
                self.state = Game.STATE_MOVEWALL
            else:
                self.send_message("Round %d starts in %d second%s." % (self.round,
                    self.countdown, "" if self.countdown == 1 else "s"))
                self.countdown -= 1
                Timer(1.0, self.start).start()
       
        if self.state == Game.STATE_MOVEWALL:
            if self.wall_offset >= Board.BOARD_LENGTH - Board.PLAY_WIDTH - 2:
                self.state = Game.STATE_CHECKWALL
            else:
                self.wall_offset += 1
                self.board.draw_gamewalls(self.wall_offset)
                Timer(0.5, self.start).start()
        
        if self.state == Game.STATE_CHECKWALL:
            self.board.update_blocks(self.blocks, remove=False)
            if self.wall_offset >= Board.BOARD_LENGTH - 2:
                # WIN
                
                # Restart
                self.state = Game.STATE_NOTPLAYING
                self.start()
            else:
                self.wall_offset += 1
                self.board.draw_gamewalls(self.wall_offset)
                Timer(0.5, self.start).start()


if __name__ == '__main__':
    # Reuse the blockplayer thread to appease OpenGL
    block_thread = None
    if 'game' in globals() and isinstance(game, Game):
        block_thread = game.block_thread
    
    game = Game()
    game.start()

