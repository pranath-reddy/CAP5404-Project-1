'''
CAP5404 Deep Learning for Computer Graphics
Project Part-1
Author: Pranath Reddy Kumbam
UFID: 8512-0977

- Run a CNN-based Connect Four Game
'''

# import libraries
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn

# define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(1, 6, 3),
            nn.ReLU(),
            nn.Conv2d(6, 12, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(72, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

# Connect 4 game class
class Connect4:
    def __init__(self, height=6, width=7):
        '''
        Args:
        height - rows in the board
        width - columns in the board
        '''
        self.height = height
        self.width = width
        # Initiate a blank board (Base state)
        self.board = [['0' for x in range(width)] for i in range(height)]

    # flatten the board to match the arramgment of the connect four UCI dataset
    def flatten(self, Board):
        return np.asarray(Board[::-1]).flatten(order='F')

    # return the current state of the board
    def return_board(self):
        return self.board

    # return a selected column
    def fetch_column(self, i):
        return [col[i] for col in self.board]

    # return a selected column for a custom board
    def fetch_column_custom(self, i, Board):
        return [col[i] for col in Board]

    # return a selected row
    def fetch_row(self, i):
        return self.board[i]

    # return all the diagonals and anti-diagonals
    def fetch_diagonals(self):
        npboard = np.asarray(self.board)
        diags = [npboard[::-1,:].diagonal(i) for i in range(-self.height,self.width)]
        diags.extend(npboard.diagonal(i) for i in range(self.height,-self.width,-1))
        return [n.tolist() for n in diags]

    # update the state of the board based on player selection
    # iterate through the rows of the selected column
    # update the deepest cell that is not blank ('0')
    def player_make_move(self, team, col):
        if '0' not in self.fetch_column(col):
            return self.board
        i = self.height - 1
        while self.board[i][col] != '0':
            i -= 1
        self.board[i][col] = team
        return self.board

    # update the state of the board for AI move
    # iterate through all the possible moves
    # select the move based on the highest prediction score
    def ai_make_move(self, B):
        scores = []
        for col in range(self.width):
            curr_state = B
            if '0' not in self.fetch_column_custom(col, curr_state):
                scores.append(-1)
                continue
            h = self.height - 1
            while curr_state[h][col] != '0':
                h -= 1
            curr_state[h][col] = '2'
            b_flat = self.flatten(curr_state)
            b_flat = np.asarray([int(a) for a in b_flat]).reshape(1,1,6,7)
            b_flat = torch.from_numpy(b_flat.astype(np.float32))
            scores.append(nn.functional.softmax(model(b_flat), dim=1).cpu().detach().numpy()[0][0])

        col = np.argmax(scores)
        if '0' not in self.fetch_column(col):
            return self.board
        i = self.height - 1
        while self.board[i][col] != '0':
            i -= 1
        self.board[i][col] = '2'
        return self.board

    # Check if player 1 or player 2 (AI) has won
    def check_win(self):
        # iterate through all the rows to check horizontal pattern match
        for i in range(self.height):
            for x in range(self.width - 3):
                if self.fetch_row(i)[x:x + 4] in [['1', '1', '1', '1'], ['2', '2', '2', '2']]:
                    return self.board[i][x]
        # iterate through all the columns to check vertical pattern match
        for i in range(self.width):
            for x in range(self.height - 3):
                if self.fetch_column(i)[x:x + 4] in [['1', '1', '1', '1'], ['2', '2', '2', '2']]:
                    return self.board[x][i]
        # Check for pattern match among all the diagonals and anti-diagonals
        for i in self.fetch_diagonals():
             if i[:4] in [['1', '1', '1', '1'], ['2', '2', '2', '2']]:
                    return i
        return None

if __name__ == "__main__":

    print("******** CNN-based Connect 4 Game ********")

    b = Connect4() # Instantiate the connect four class
    # load model and push to device
    model = torch.load('Weights/CNN_Connect4_Unbalanced.pth', map_location=torch.device('cpu'))

    # Play the game as described in the report (Figure-)
    while True:
        for i in b.return_board():
            print(i)
        if b.check_win() != None:
            print('AI Wins!')
            break
        col = int(input('Player 1 choose column: '))-1
        b.player_make_move('1' , col)
        for i in b.return_board():
            print(i)
        if b.check_win() != None:
            print('Player 1 Wins!')
            break
        board_copy = []
        for i in b.return_board():
            board_copy.append(i)
        b.ai_make_move(np.asarray(board_copy))
        print('AI (Player 2) makes move')

        # Check for draw
        tick = 0
        for i in range(0, len(b.return_board())):
            if '0' not in b.return_board()[i]:
                tick += 1
        if tick == 6:
            print('Draw')
            break
