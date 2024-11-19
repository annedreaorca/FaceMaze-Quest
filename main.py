import cv2
import numpy as np
import pyautogui as gui
import time
import winsound 
from random import randint

gui.PAUSE = 0

model_path = './model/face_model.caffemodel'
prototxt_path = './model/face_model_config.prototxt'

class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.visited = False
        self.walls = {'top': True, 'right': True, 'bottom': True, 'left': True}

    def check_neighbors(self, cols, rows, grid_cells):
        neighbors = []
        index = lambda x, y: x + y * cols
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Left, right, top, bottom
        for dx, dy in directions:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                neighbor = grid_cells[index(nx, ny)]
                if not neighbor.visited:
                    neighbors.append(neighbor)
        return neighbors[randint(0, len(neighbors) - 1)] if neighbors else None

class Maze:
    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows
        self.grid_cells = [Cell(x, y) for y in range(rows) for x in range(cols)]

    def remove_walls(self, current, next):
        dx = current.x - next.x
        if dx == 1:
            current.walls['left'] = False
            next.walls['right'] = False
        elif dx == -1:
            current.walls['right'] = False
            next.walls['left'] = False
        dy = current.y - next.y
        if dy == 1:
            current.walls['top'] = False
            next.walls['bottom'] = False
        elif dy == -1:
            current.walls['bottom'] = False
            next.walls['top'] = False

    def generate_maze(self):
        current_cell = self.grid_cells[0]
        stack = []
        break_count = 1
        while break_count != len(self.grid_cells):
            current_cell.visited = True
            next_cell = current_cell.check_neighbors(self.cols, self.rows, self.grid_cells)
            if next_cell:
                next_cell.visited = True
                break_count += 1
                stack.append(current_cell)
                self.remove_walls(current_cell, next_cell)
                current_cell = next_cell
            elif stack:
                current_cell = stack.pop()
        return self.grid_cells

# Dynamic maze creation
maze_cols, maze_rows = 10, 10
maze_obj = Maze(maze_cols, maze_rows)
grid_cells = maze_obj.generate_maze()

# Convert the maze grid into a 2D list for FaceMaze Quest logic
maze = [[1 for _ in range(maze_cols)] for _ in range(maze_rows)]
for cell in grid_cells:
    maze[cell.y][cell.x] = 0

# Initialize other game elements
player_position = [0, 0]  # Starting position
finish_position = [maze_rows - 1, maze_cols - 1]  # Goal position
player_path = [player_position.copy()]
start_time = time.time()
maze_completed = False

player_position = [0, 0] # Starting position
player_path = [player_position.copy()]  # To store the player's path

start_time = time.time()  # To track the start time of the game
maze_completed = False

def play_sound(frequency=1000, duration=200):
    winsound.Beep(frequency, duration) 

def detect(net, frame):
    detected_faces = []
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            detected_faces.append({'start': (startX, startY), 'end': (endX, endY), 'confidence': confidence})
    return detected_faces

def drawFace(frame, detected_faces):
    for face in detected_faces:
        cv2.rectangle(frame, face['start'], face['end'], (0, 255, 0), 10)
    return frame

def checkRect(detected_faces, bbox):
    for face in detected_faces:
        x1, y1 = face['start']
        x2, y2 = face['end']
        if x1 > bbox[0] and x2 < bbox[1]:
            if y1 > bbox[3] and y2 < bbox[2]:
                return True
    return False

def move_player(direction):
    global player_position
    x, y = player_position
    current_cell = grid_cells[x * maze_cols + y]

    if direction == 'up' and not current_cell.walls['top']:
        player_position[0] -= 1
    elif direction == 'down' and not current_cell.walls['bottom']:
        player_position[0] += 1
    elif direction == 'left' and not current_cell.walls['left']:
        player_position[1] -= 1
    elif direction == 'right' and not current_cell.walls['right']:
        player_position[1] += 1

    player_path.append(player_position.copy())  # Track the player's path

def move(detected_faces, bbox):
    global last_mov
    for face in detected_faces:
        x1, y1 = face['start']
        x2, y2 = face['end']
        if checkRect(detected_faces, bbox):
            last_mov = 'center'
            return
        elif last_mov == 'center':
            if x1 < bbox[0]:
                move_player('left')
                play_sound()  
                last_mov = 'left'
            elif x2 > bbox[1]:
                move_player('right')
                play_sound()  
                last_mov = 'right'
            if y2 > bbox[2]:
                move_player('down')
                play_sound()
                last_mov = 'down'
            elif y1 < bbox[3]:
                move_player('up')
                play_sound() 
                last_mov = 'up'

finish_position = [9, 9]

maze_completed = False

import math

# Global variable to store the current rotation angle of the character
current_angle = 0

import numpy as np
import cv2

def draw_maze():
    cell_size = 60
    maze_frame = np.zeros((maze_rows * cell_size, maze_cols * cell_size, 3), dtype=np.uint8)  # Black background

    # Load assets
    cell_img = cv2.imread('./assets/cell.png')
    hline_img = cv2.imread('./assets/box.png')
    vline_img = cv2.imread('./assets/box.png')
    character_img = cv2.imread('./assets/character.png', cv2.IMREAD_UNCHANGED)  # Load the player character image
    start_img = cv2.imread('./assets/start.png', cv2.IMREAD_UNCHANGED)  # Load the start position image
    goal_img = cv2.imread('./assets/goal.png', cv2.IMREAD_UNCHANGED)  # Load the goal image

    # Resize assets to fit wall/cell sizes
    cell_img = cv2.resize(cell_img, (cell_size, cell_size))
    hline_img = cv2.resize(hline_img, (cell_size, int(cell_size / 15)))
    vline_img = cv2.resize(vline_img, (int(cell_size / 15), cell_size))
    start_img = cv2.resize(start_img, (cell_size, cell_size))  # Resize start image to fit a cell
    goal_img = cv2.resize(goal_img, (cell_size, cell_size))  # Resize goal image to fit a cell

    # Resize the character image to be smaller than the cell
    char_scale = 0.6 # Scale factor for the character
    char_width = int(cell_size * char_scale)
    char_height = int(cell_size * char_scale)
    character_img = cv2.resize(character_img, (char_width, char_height))  # Resize character image

    def blend_image(background, overlay, top_left):
        """Blends an overlay image with transparency onto a background."""
        x, y = top_left
        h, w = overlay.shape[:2]
        alpha = overlay[:, :, 3] / 255.0  # Extract alpha channel for blending
        for c in range(3):  # Blend each color channel
            if (y + h <= background.shape[0]) and (x + w <= background.shape[1]):  # Ensure the overlay fits
                background[y:y + h, x:x + w, c] = (
                    alpha * overlay[:, :, c] + (1 - alpha) * background[y:y + h, x:x + w, c]
                )
        return background

    # Draw the maze cells and walls
    for cell in grid_cells:
        x, y = cell.x, cell.y
        top_left_x, top_left_y = x * cell_size, y * cell_size

        # Add the cell base image
        maze_frame[top_left_y:top_left_y + cell_size, top_left_x:top_left_x + cell_size] = cell_img

        # Draw walls with assets
        if cell.walls['top']:  # Top wall
            maze_frame[top_left_y:top_left_y + hline_img.shape[0], top_left_x:top_left_x + cell_size] = hline_img
        if cell.walls['right']:  # Right wall
            maze_frame[top_left_y:top_left_y + cell_size, top_left_x + cell_size - vline_img.shape[1]:top_left_x + cell_size] = vline_img
        if cell.walls['bottom']:  # Bottom wall
            maze_frame[top_left_y + cell_size - hline_img.shape[0]:top_left_y + cell_size, top_left_x:top_left_x + cell_size] = hline_img
        if cell.walls['left']:  # Left wall
            maze_frame[top_left_y:top_left_y + cell_size, top_left_x:top_left_x + vline_img.shape[1]] = vline_img

    # Draw the starting position
    sx, sy = player_path[0]  # Starting position
    start_top_left_y, start_top_left_x = sx * cell_size, sy * cell_size
    blend_image(maze_frame, start_img, (start_top_left_x, start_top_left_y))

    # Draw the player's current position
    px, py = player_position
    player_top_left_y = px * cell_size + (cell_size - char_height) // 2
    player_top_left_x = py * cell_size + (cell_size - char_width) // 2
    blend_image(maze_frame, character_img, (player_top_left_x, player_top_left_y))

    # Draw the finish position (goal) with the goal image
    fx, fy = finish_position
    goal_top_left_y = fx * cell_size
    goal_top_left_x = fy * cell_size
    blend_image(maze_frame, goal_img, (goal_top_left_x, goal_top_left_y))

    if maze_completed:
        cv2.putText(maze_frame, "Maze Completed!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

    return maze_frame

# Check if player has reached the finish position after each move
def check_finish():
    global maze_completed
    if player_position == finish_position:
        maze_completed = True
        end_time = time.time()
        time_taken = int(end_time - start_time)
        print(f"Maze completed in {time_taken} seconds!")
        play_sound(frequency=1500, duration=500)

# Update the play loop to check for finish after moving the player
def play(prototxt_path, model_path):
    global last_mov
    last_mov = ''
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    cap = cv2.VideoCapture(0)

    while not cap.isOpened():
        cap = cv2.VideoCapture(0)

    # Co-ordinates of the bounding box on frame
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    box_width, box_height = 250, 250 
    left_x, top_y = frame_width // 2 - box_width // 2, frame_height // 2 - box_height // 2
    right_x, bottom_y = frame_width // 2 + box_width // 2, frame_height // 2 + box_height // 2
    bbox = [left_x, right_x, bottom_y, top_y]

    while True:
        ret, frame = cap.read()
        if not ret:
            return 0

        frame = cv2.flip(frame, 1)
        detected_faces = detect(net, frame)
        frame = drawFace(frame, detected_faces)
        frame = cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), (0, 0, 255), 5)

        # Only move if the maze is not completed
        if not maze_completed:
            move(detected_faces, bbox)
            check_finish()  # Check if the player reached the finish

        # Update and display the maze view
        maze_view = draw_maze()
        cv2.imshow('maze_view', maze_view)

        # Display the camera feed
        cv2.imshow('camera_feed', frame)
        if cv2.waitKey(5) == 27:  # ESC key to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play(prototxt_path, model_path)