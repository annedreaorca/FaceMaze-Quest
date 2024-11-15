import cv2
import numpy as np
import pyautogui as gui
import time
import winsound 

gui.PAUSE = 0

model_path = './model/face_model.caffemodel'
prototxt_path = './model/face_model_config.prototxt'

maze = [
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 0, 1, 1, 1, 1, 1, 0, 0]
]

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
    if direction == 'up' and maze[x - 1][y] == 0:
        player_position[0] -= 1
    elif direction == 'down' and maze[x + 1][y] == 0:
        player_position[0] += 1
    elif direction == 'left' and maze[x][y - 1] == 0:
        player_position[1] -= 1
    elif direction == 'right' and maze[x][y + 1] == 0:
        player_position[1] += 1
    player_path.append(player_position.copy())  # Track the new path

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

def draw_maze():
    cell_size = 60
    maze_frame = np.ones((len(maze) * cell_size, len(maze[0]) * cell_size, 3), dtype=np.uint8) * 255  # White background

    for i in range(len(maze)):
        for j in range(len(maze[0])):
            # Colors for player and finish
            player_color = (0, 255, 0)  # Green for player
            finish_color = (255, 0, 0)  # Red for finish
            line_color = (0, 0, 0)  # Black for lines

            # Draw grid cell walls based on maze layout
            top_left = (j * cell_size, i * cell_size)
            bottom_right = ((j + 1) * cell_size, (i + 1) * cell_size)

            if maze[i][j] == 1:
                # Draw borders around each cell to form the maze structure
                cv2.rectangle(maze_frame, top_left, bottom_right, line_color, 1)  # Outline of the cell

    # Draw the player's current position
    px, py = player_position
    player_top_left = (py * cell_size, px * cell_size)
    player_bottom_right = (py * cell_size + cell_size, px * cell_size + cell_size)
    cv2.rectangle(maze_frame, player_top_left, player_bottom_right, player_color, -1)

    # Draw the finish position
    fx, fy = finish_position
    finish_top_left = (fy * cell_size, fx * cell_size)
    finish_bottom_right = (fy * cell_size + cell_size, fx * cell_size + cell_size)
    cv2.rectangle(maze_frame, finish_top_left, finish_bottom_right, finish_color, -1)

    # Overlay maze completion message
    if maze_completed:
        cv2.putText(maze_frame, "Maze Completed!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 3)
    
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