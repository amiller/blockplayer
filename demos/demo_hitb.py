import struct
import random
import os.path
import signal
from math import ceil
from glob import glob
from threading import Timer, Thread, Lock

import zmq
import numpy as np

from blockplayer import main
from blockplayer import stencil
from blockplayer import config
from blockplayer import blockcraft
from blockplayer import blockdraw
from blockplayer import dataset
import glxcontext
import opennpy

FOR_REAL = True

class HitBException(Exception):
    pass


if 'socket' in globals():
    socket.close()

if 'context' not in globals():
    context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.connect('tcp://*:8134')

cmd_lock = Lock()
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

def build_tntprimed(x, y, z, ticks):
    return 't' + struct.pack(">hhhh", x, y, z, ticks)

def send_globalqueue():
    global cmd_queue, commands
    
    cmd_lock.acquire()
    socket.send(struct.pack('>i', commands) + cmd_queue)
    
    cmd_queue = ''
    commands = 0
    cmd_lock.release()

def add_queue(command):
    global cmd_queue, commands
    
    cmd_lock.acquire()
    cmd_queue += command
    commands += 1
    cmd_lock.release()


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
    TNT = 46
    FENCE = 85


class Board:
    PLAY_WIDTH = 9
    PLAY_HEIGHT = 9
    
    # The width of the border from x=0 to the playing area
    # This is made up of the grass and purple borders
    BOARD_BORDER = 2
    
    BOARD_LENGTH = 40
    BOARD_WIDTH = 12
    BOARD_HEIGHT = PLAY_HEIGHT
    
    # No block present, no design present
    MATCH_BLANK = -1
    # Block present, design present
    MATCH_MATCH = 0
    # No block present, design present
    MATCH_EMPTY = 1
    # Block present, no design present
    MATCH_COLLIDE = 2
    
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
    
    def __init__(self, tracks=True):
        self.wall_offset = 0
        self.wall_abs = Board.BOARD_LENGTH
        self.tracks = tracks
        
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
    
    def set_tntprimed(self, x, y, z, ticks):
        add_queue(build_tntprimed(x, y, z, ticks))
    
    def setup_gameboard(self):
        """Builds the game board and sends it to the server"""
        
        # Clear the board
        for x in xrange(Board.BOARD_LENGTH+1):
            for z in xrange(Board.BOARD_WIDTH+3):
                for y in xrange(Board.BOARD_HEIGHT+1):
                    self.set_block(x, y, z, Material.AIR)
                    self.set_block(z, y, x, Material.AIR)
        
        # First, the perimeter of grass
        for x in xrange(Board.BOARD_LENGTH+1):
            self.set_block(x, -1, 0, Material.GRASS)
            self.set_block(0, -1, x, Material.GRASS)
        for x in xrange(Board.BOARD_LENGTH-Board.BOARD_WIDTH+1):
            self.set_block(Board.BOARD_WIDTH+x, -1, Board.BOARD_WIDTH, Material.GRASS)
            self.set_block(Board.BOARD_WIDTH, -1, Board.BOARD_WIDTH+x, Material.GRASS)
        
        # Next, the purple border
        for x in xrange(Board.BOARD_LENGTH):
            self.set_wool(x+1, -1, 1, DyeColor.PURPLE)
            self.set_wool(1, -1, x+1, DyeColor.PURPLE)
        for x in xrange(Board.BOARD_LENGTH-Board.BOARD_WIDTH+2):
            self.set_wool(Board.BOARD_WIDTH+x-1, -1, Board.BOARD_WIDTH-1, DyeColor.PURPLE)
            self.set_wool(Board.BOARD_WIDTH-1, -1, Board.BOARD_WIDTH+x-1, DyeColor.PURPLE)
        
        # The fence border
        for x in xrange(Board.BOARD_LENGTH+1):
            self.set_block(x, 0, 0, Material.FENCE)
            self.set_block(0, 0, x, Material.FENCE)
        for x in xrange(Board.BOARD_LENGTH-Board.BOARD_WIDTH+1):
            self.set_block(Board.BOARD_WIDTH+x, 0, Board.BOARD_WIDTH, Material.FENCE)
            self.set_block(Board.BOARD_WIDTH, 0, Board.BOARD_WIDTH+x, Material.FENCE)
        
        # Clear the origin
        self.set_block(0, 0, 0, Material.AIR)
        # Add a block for the player to jump on to get over the fences
        self.set_wool(1, 0, 1, DyeColor.PURPLE)
        
        # The playing board
        for x in xrange(Board.BOARD_BORDER, Board.BOARD_LENGTH+1):
            for z in xrange(Board.BOARD_BORDER, Board.BOARD_WIDTH-Board.BOARD_BORDER+1):
                self.set_wool(x, -1, z, Board.TRACKS[z-Board.BOARD_BORDER]
                    if self.tracks else DyeColor.BLACK)
                self.set_wool(z, -1, x, Board.TRACKS[z-Board.BOARD_BORDER]
                    if self.tracks else DyeColor.BLACK)
        
        # The blue playing area square marker
        for x in xrange(Board.BOARD_BORDER, Board.PLAY_WIDTH+Board.BOARD_BORDER+1):
            self.set_wool(x, -1, Board.PLAY_WIDTH+Board.BOARD_BORDER, DyeColor.BLUE)
            self.set_wool(Board.PLAY_WIDTH+Board.BOARD_BORDER, -1, x, DyeColor.BLUE)
            
            self.set_wool(x, -1, Board.BOARD_BORDER-1, DyeColor.BLUE)
            self.set_wool(Board.BOARD_BORDER-1, -1, x, DyeColor.BLUE)
        
        # The glass ceiling
        for x in xrange(Board.BOARD_LENGTH+1):
            for z in xrange(Board.BOARD_WIDTH+1):
                self.set_block(x, Board.BOARD_HEIGHT+1, z, Material.GLASS)
                self.set_block(z, Board.BOARD_HEIGHT+1, x, Material.GLASS)
        
        # Draw the blank game walls
        self.draw_gamewalls(0, True)
        
        # Set the bounds
        # The extra 2 covers the ground below the origin and the glass ceiling
        # The extra 1 for the width covers the animation display area
        self.add_bound(0, -1, 0, Board.BOARD_LENGTH, Board.BOARD_WIDTH+1,
            Board.BOARD_HEIGHT+2)
        self.add_bound(0, -1, 0, Board.BOARD_WIDTH, Board.BOARD_LENGTH+1,
            Board.BOARD_HEIGHT+2)
        
        # Flush the command queue
        send_globalqueue()
    
    def draw_gamewalls(self, offset=0, blank=False, blocks=None):
        """Draws the walls where the designs appear at an offset from the end of
        the board. If blank is True, the designs are not drawn. If blocks is not
        None, it should be blocks in the playing area not to clear."""
        
        # Clear the old walls
        if offset != self.wall_offset:
            # Starts at 1 so it draws the walls over the purple borders
            for x in xrange(1, Board.BOARD_WIDTH-1+1):
                for y in xrange(Board.BOARD_HEIGHT+1):
                    clear_x = True
                    clear_z = True
                    if blocks is not None:
                        if self.wall_abs-Board.BOARD_BORDER < Board.PLAY_WIDTH and \
                            (0 <= x - Board.BOARD_BORDER < Board.PLAY_WIDTH 
                            and y < Board.PLAY_HEIGHT):
                            if blocks[self.wall_abs-Board.BOARD_BORDER][y][x-Board.BOARD_BORDER]:
                                clear_x = False
                            if blocks[x-Board.BOARD_BORDER][y][self.wall_abs-Board.BOARD_BORDER]:
                                clear_z = False
                    
                    # x-side (left from origin) wall
                    if clear_x:
                        self.set_block(Board.BOARD_LENGTH-self.wall_offset, y,
                            x, Material.AIR)
                    
                    # z-side (right from origin) wall
                    if clear_z:
                        self.set_block(x, y, Board.BOARD_LENGTH-self.wall_offset,
                            Material.AIR)
        
        # Absolute position of wall from origin
        wall_abs = Board.BOARD_LENGTH-offset
        
        # Setup blank yellow walls
        for x in xrange(1, Board.BOARD_WIDTH-1+1):
            for y in xrange(Board.BOARD_HEIGHT+1):
                # x-side (left from origin) wall
                self.set_wool(wall_abs, y, x, DyeColor.YELLOW)
                
                # z-side (right from origin) wall
                self.set_wool(x, y, wall_abs, DyeColor.YELLOW)
        
        # Clear away any design
        if not blank:
            if self.x_design is not None:
                for x,col in enumerate(self.x_design):
                    for y,isset in enumerate(col):
                        if isset:
                            if blocks is not None:
                                if self.wall_abs-Board.BOARD_BORDER < Board.PLAY_WIDTH and \
                                    (0 <= x - Board.BOARD_BORDER < Board.PLAY_WIDTH 
                                    and y < Board.PLAY_HEIGHT):
                                    if blocks[self.wall_abs-Board.BOARD_BORDER][y][x-Board.BOARD_BORDER]:
                                        continue
                            self.set_block(wall_abs, y, x+Board.BOARD_BORDER,
                                Material.AIR)
            
            if self.z_design is not None:
                for x,col in enumerate(self.z_design):
                    for y,isset in enumerate(col):
                        if isset:
                            if blocks is not None:
                                if self.wall_abs-Board.BOARD_BORDER < Board.PLAY_WIDTH and \
                                    (0 <= x - Board.BOARD_BORDER < Board.PLAY_WIDTH 
                                    and y < Board.PLAY_HEIGHT):
                                    if blocks[x-Board.BOARD_BORDER][y][self.wall_abs-Board.BOARD_BORDER]:
                                        continue
                            self.set_block(x+Board.BOARD_BORDER, y, wall_abs,
                                Material.AIR)
        
        self.wall_offset = offset
        self.wall_abs = wall_abs
        
        send_globalqueue()
    
    def load_design(self, path):
        """Load in the x and z wall designs from the file at `path`"""
        with open(path, 'r') as fp:
            lines = fp.readlines(Board.PLAY_WIDTH)
            # Two board designs separated by a blank line
            if len(lines) != Board.PLAY_HEIGHT*2 + 1:
                raise HitBException("Cutout design file \"%s\" incorrectly formatted." % path)
            
            # Create the empty design lists
            self.x_design = []
            self.z_design = []
            for x in xrange(Board.PLAY_WIDTH):
                self.x_design.append([False]*Board.PLAY_HEIGHT)
                self.z_design.append([False]*Board.PLAY_HEIGHT)
            
            # Fill the x-side design
            for y,line in enumerate(lines[:Board.PLAY_HEIGHT]):
                for x,c in enumerate(line):
                    if c == '\n': break
                    if x >= Board.PLAY_WIDTH: break
                    self.x_design[x][Board.PLAY_HEIGHT-y-1] = (c == '#')
            
            # Fill the z-side design
            for y,line in enumerate(lines[Board.PLAY_HEIGHT+1:]):
                for x,c in enumerate(line):
                    if c == '\n': break
                    if x >= Board.PLAY_WIDTH: break
                    self.z_design[x][Board.PLAY_HEIGHT-y-1] = (c == '#')
    
    def clear_design(self):
        """Clears both wall designs"""
        self.x_design = []
        self.z_design = []
    
    def clear_playarea(self):
        """Clears the entire playing area"""
        for x in xrange(Board.PLAY_WIDTH):
            for y in xrange(Board.PLAY_HEIGHT):
                for z in xrange(Board.PLAY_WIDTH):
                    self.set_block(x+Board.BOARD_BORDER, y,
                        z+Board.BOARD_BORDER, Material.AIR)
        send_globalqueue()
    
    def update_blocks(self, blocks, remove=True, glass=False):
        """\
        Draw the blocks in the playing area. The blocks array must be of size
        PLAY_WIDTH x PLAY_WIDTH x PLAY_HEIGHT. If remove is True, it clears
        any block with a False value in `blocks`
        """
        
        if blocks is None:
            raise HitBException("Invalid blocks array passed to update function. Cannot update blocks.")
        if blocks.shape != (Board.PLAY_WIDTH, Board.PLAY_HEIGHT, Board.PLAY_WIDTH):
            raise HitBException("Blocks array has wrong dimensions. Cannot update blocks.")
        
        for x in xrange(Board.PLAY_WIDTH):
            for y in xrange(Board.PLAY_HEIGHT):
                for z in xrange(Board.PLAY_WIDTH):
                    if blocks[x,y,z]:
                        if not glass:
                            self.set_wool(x+Board.BOARD_BORDER, y,
                                z+Board.BOARD_BORDER, DyeColor.LIGHT_BLUE)
                        else:
                            self.set_block(x+Board.BOARD_BORDER, y,
                                z+Board.BOARD_BORDER, Material.GLASS)
                    elif remove:
                        self.set_block(x+Board.BOARD_BORDER, y,
                            z+Board.BOARD_BORDER, Material.AIR)
        
        send_globalqueue()
    
    def _match_block(self, blocks, design, x, y, bx, by, bz):
        """Match a single block and return a Board.MATCH constant"""
        # No design present
        if not design[y][x]:
            # Block present
            if blocks[bx][by][bz]:
                return Board.MATCH_COLLIDE
            # No block present
            else:
                return Board.MATCH_BLANK
        
        # Design present
        else:
            # Block present
            if blocks[bx][by][bz]:
                return Board.MATCH_MATCH
            # No block present
            else:
                return Board.MATCH_EMPTY
                
    
    def match_blocks(self, blocks, wall_abs=None):
        """\
        Matches blocks to the loaded design on both walls. `wall_abs` should
        be an absolute offset from the origin to check. Returns a 2-tuple
        containing 2d lists representing the blocks matched. For each block,
        a value of Board.MATCH constant is assigned.
        """
        
        if blocks is None:
            return None
        if blocks.shape != (Board.PLAY_WIDTH, Board.PLAY_HEIGHT, Board.PLAY_WIDTH):
            return None
        
        if wall_abs is None:
            wall_abs = self.wall_abs
        
        # Correct abs offset for the border
        blocks_idx = wall_abs - Board.BOARD_BORDER
        
        # Populate the empty match matrices
        x_match = []
        z_match = []
        for x in xrange(Board.PLAY_WIDTH):
            x_match.append([Board.MATCH_BLANK]*Board.PLAY_HEIGHT)
            z_match.append([Board.MATCH_BLANK]*Board.PLAY_HEIGHT)
        
        for x in xrange(Board.PLAY_WIDTH):
            for y in xrange(Board.PLAY_HEIGHT):
                # This is very fragile. It works, so I won't touch it.
                x_match[y][x] = self._match_block(blocks, self.x_design, x, y,
                    blocks_idx, x, y)
                z_match[x][y] = self._match_block(blocks, self.z_design,
                    y, x, x, y, blocks_idx)
        
        return (x_match, z_match)
    
    def display(self, disp):
        """Display a block image string"""
        xlen = Board.BOARD_LENGTH*2 - Board.BOARD_WIDTH*2
        for x in xrange(xlen):
            for y in xrange(Board.PLAY_HEIGHT):
                if x < Board.BOARD_LENGTH - Board.BOARD_WIDTH:
                    x_coord = Board.BOARD_LENGTH - x
                    z_coord = Board.BOARD_WIDTH + 1
                else:
                    x_coord = Board.BOARD_WIDTH + 1
                    z_coord = x - Board.BOARD_LENGTH + Board.BOARD_WIDTH*2 + 2
                
                if disp[x,y]:
                    #self.set_block(x_coord, y, z_coord, Material.TNT)
                    self.set_tntprimed(x_coord, y, z_coord, 5)
                else:
                    self.set_block(x_coord, y, z_coord, Material.AIR)


class Game:
    STATE_QUIT = -1
    STATE_NOTPLAYING = 0
    STATE_LOADDESIGN = 1
    STATE_COUNTDOWN = 2
    STATE_MOVEWALL = 3
    STATE_CHECKWALL = 4
    STATE_ENDROUND = 5
    
    # The rate at which the wall moves
    WALL_FPS = 2
    WALL_TICK = 1.0/WALL_FPS
    
    # The time between rounds
    AFTER_ROUND_TIME = 3
    
    WIN_POINTS = 10
    LOSE_POINTS = 0
    
    def __init__(self, clear_bounds=True, tracks=True):
        self.restart(clear_bounds, tracks)
    
    def restart(self, clear_bounds=True, tracks=True):
        """Sets all variables to their initial state, redraws the game board,
        and restarts blockplayer."""
        
        self.state = Game.STATE_NOTPLAYING
        self.countdown = 5
        self.round = 0
        self.score = 0
        self.wall_offset = 0
        
        self.x_match = None
        self.z_match = None
        
        self.block_initialized = False
        self.update_blocks = True
        self.block_loop_quit = False
        self.blocks = None
        self.block_thread = Thread(target=self.block_setup)
        
        self.board = Board(tracks=tracks)
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
        glxcontext.makecurrent()
        main.initialize()
        
        config.load('data/newest_calibration')
        opennpy.align_depth_to_rgb()
        dataset.setup_opencl()
        
        self.blocks = None
        self.block_loop_quit = False
        self.block_initialized = True
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
        if self.state < Game.STATE_CHECKWALL and self.update_blocks:
            self.blocks = blockcraft.translated_rotated(main.R_correct, main.grid.occ)
            self.blocks = self.block_slice(self.blocks)
            self.board.update_blocks(self.blocks,
                remove=self.board.wall_abs-Board.BOARD_BORDER > Board.PLAY_WIDTH)
    
    def block_loop(self):
        """Loops the blockplayer frame processing function"""
        while not self.block_loop_quit:
            self.block_once()
    
    def print_score(self):
        """Display the player's score"""
        self.send_message("Score: %d" % self.score)
    
    def start(self):
        """Begins the game."""
        
        # Block until blockplayer is initialized
        while not self.block_initialized:
            pass
        
        if self.state == Game.STATE_QUIT:
            return
        
        if self.state == Game.STATE_NOTPLAYING:
            self.countdown = 5
            self.round += 1
            self.wall_offset = 0
            
            self.update_blocks = True
            self.board.clear_design()
            
            self.x_match = None
            self.z_match = None
            
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
            if self.board.wall_abs - Board.BOARD_BORDER < Board.PLAY_WIDTH:
                self.state = Game.STATE_CHECKWALL
                self.update_blocks = False
            else:
                self.wall_offset += 1
                self.board.draw_gamewalls(self.wall_offset)
                Timer(Game.WALL_TICK, self.start).start()
        
        if self.state == Game.STATE_CHECKWALL:
            self.board.update_blocks(self.blocks, remove=False)
            
            # Match the blocks to the designs
            x_match,z_match = self.board.match_blocks(self.blocks)
            
            # Check for collisions
            collide = False
            for matches in (x_match, z_match):
                for x,col in enumerate(matches):
                    for y,match in enumerate(col):
                        if match == Board.MATCH_COLLIDE:
                            collide = True
                            break
            
            if collide:
                # Collision detected. Set the colliding blocks to red wool, and
                # the rest of the blocks to glass for visibility
                for x in xrange(Board.PLAY_WIDTH):
                    for y in xrange(Board.PLAY_HEIGHT):
                        for z in xrange(Board.PLAY_WIDTH):
                            if self.blocks[x,y,z]:
                                if x == self.board.wall_abs-Board.BOARD_BORDER \
                                    and x_match[z][y] == Board.MATCH_COLLIDE:
                                    self.board.set_wool(x+Board.BOARD_BORDER, y,
                                         z+Board.BOARD_BORDER, DyeColor.RED)
                                
                                elif z == self.board.wall_abs-Board.BOARD_BORDER \
                                    and z_match[x][y] == Board.MATCH_COLLIDE:
                                    self.board.set_wool(x+Board.BOARD_BORDER, y,
                                         z+Board.BOARD_BORDER, DyeColor.RED)
                                else:
                                    self.board.set_block(x+Board.BOARD_BORDER, y,
                                        z+Board.BOARD_BORDER, Material.GLASS)
                            else:
                                self.board.set_block(x+Board.BOARD_BORDER, y,
                                    z+Board.BOARD_BORDER, Material.AIR)
                
                # Flush the command queue
                send_globalqueue()
                
                self.send_message("Your blocks collided with the wall.")
                self.send_message("You lost this round :(")
                
                self.score += Game.LOSE_POINTS
                
                self.state = Game.STATE_ENDROUND
            else:
                if self.x_match is None: self.x_match = x_match
                if self.z_match is None: self.z_match = z_match
                
                for x in xrange(Board.PLAY_WIDTH):
                    for y in xrange(Board.PLAY_HEIGHT):
                        if self.x_match[x][y] == Board.MATCH_EMPTY \
                            and x_match[x][y] == Board.MATCH_MATCH:
                            self.x_match[x][y] = Board.MATCH_MATCH
                        
                        if self.z_match[x][y] == Board.MATCH_EMPTY \
                            and z_match[x][y] == Board.MATCH_MATCH:
                            self.z_match[x][y] = Board.MATCH_MATCH
                
                if self.board.wall_abs - Board.BOARD_BORDER <= 0:
                    satisfied = True
                    for x in xrange(Board.PLAY_WIDTH):
                        if not satisfied:
                            break
                        for y in xrange(Board.PLAY_HEIGHT):
                            if self.x_match[x][y] == Board.MATCH_EMPTY or \
                                self.z_match[x][y] == Board.MATCH_EMPTY:
                                satisfied = False
                                break
                    
                    if satisfied:
                        self.send_message("You won this round!")
                        self.score += Game.WIN_POINTS
                    else:
                        self.send_message("Your blocks did not satisfy both designs.")
                        self.send_message("You lost this round :(")
                        self.score += Game.LOSE_POINTS
                    
                    self.state = Game.STATE_ENDROUND
                else:
                    self.wall_offset += 1
                    self.board.draw_gamewalls(self.wall_offset, blocks=self.blocks)
                    Timer(Game.WALL_TICK, self.start).start()
        
        if self.state == Game.STATE_ENDROUND:
            self.print_score()
            
            # Reset the game wall
            self.board.draw_gamewalls(blank=True, blocks=self.blocks)
            
            self.update_blocks = False
            self.state = Game.STATE_NOTPLAYING
            Timer(Game.AFTER_ROUND_TIME, self.start).start()


def disp_format(anim):
    """Formats an animation display string to be passed to the display"""
    n = np.array([c=='#' for c in ''.join([line[::-1] for line in anim.split('\n')[::-1]])])
    n = np.split(n, Board.BOARD_HEIGHT)
    return np.rot90(n)

WIN_TEXT = disp_format("""\
..........................................................
............#........#..#######..#.....#..................
............#........#.....#.....##....#..................
............#........#.....#.....#.#...#..................
............#........#.....#.....#..#..#..................
............#...##...#.....#.....#...#.#..................
............#..#..#..#.....#.....#....##..................
............###....###..#######..#.....#..................
..........................................................""")


# Debugging functions
MATCH_CHARS = {
    Board.MATCH_BLANK: '.',
    Board.MATCH_MATCH: '#',
    Board.MATCH_EMPTY: ' ',
    Board.MATCH_COLLIDE: 'X'
}
def match_to_string(match):
    rows = ['' for x in xrange(len(match[0]))]
    for col in match:
        for y,m in enumerate(col[::-1]):
            rows[y] += MATCH_CHARS[m]
    return '\n'.join(rows)

def block_slice_to_string(blocks, x=None, z=None):
    if blocks is None:
        return None
    if x is None and z is None:
        return None
    if x is not None:
        s = ''
        for cy in xrange(Board.PLAY_HEIGHT-1, 0-1, -1):
            for cz in xrange(Board.PLAY_WIDTH):
                s += '#' if blocks[x][cy][cz] else '.'
            s += '\n'
        return s
    else:
        s = ''
        for cy in xrange(Board.PLAY_HEIGHT-1, 0-1, -1):
            for cx in xrange(Board.PLAY_WIDTH):
                s += '#' if blocks[cx][cy][z] else '.'
            s += '\n'
        return s


if __name__ == '__main__':
    import sys
    
    def usage():
        print "Usage: %s [-notracks]" % os.path.basename(sys.argv[0])
        sys.exit(2)
    
    if len(sys.argv) > 2:
        usage()
    
    tracks = True
    if len(sys.argv) == 2:
        if sys.argv[1] != '-notracks':
            usage()
        tracks = False
    
    # Handle KeyboardInterrupts
    def handle_sigint(sig, stack):
        print "Quitting game."
        game.quit()
    signal.signal(signal.SIGINT, handle_sigint)
    
    game = Game(tracks=tracks)
    game.start()

