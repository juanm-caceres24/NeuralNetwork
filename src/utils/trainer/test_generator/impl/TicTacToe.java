package src.utils.trainer.test_generator.impl;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import src.utils.trainer.test_generator.TestGenerator;

public class TicTacToe implements TestGenerator {

    private static final int BOARD_SIZE = 9;
    private static final int MAX_PIECES = 3;
    private static final double OWN_PIECE = 1.0;
    private static final double ENEMY_PIECE = -1.0;
    private static final double EMPTY = 0.0;
    private static final double WIN = 1.0;
    private static final double MOVE_FROM = -1.0;
    private static final int[][] WINNING_LINES = {
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8},
        {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
        {0, 4, 8}, {2, 4, 6}
    };

    @Override
    public double[][][] generateTrainingData() {
        List<double[][]> samples = new ArrayList<>();
        for (int ownMask = 0; ownMask < 1 << BOARD_SIZE; ownMask++) {
            int ownPieces = Integer.bitCount(ownMask);
            if (ownPieces > MAX_PIECES) continue;
            for (int enemyMask = 0; enemyMask < 1 << BOARD_SIZE; enemyMask++) {
                int enemyPieces = Integer.bitCount(enemyMask);
                if ((ownMask & enemyMask) != 0 || enemyPieces > MAX_PIECES) continue;
                if (enemyPieces != ownPieces && enemyPieces != ownPieces + 1) continue;
                double[] board = createBoard(ownMask, enemyMask);
                samples.add(new double[][] { board, chooseMove(board) });
            }
        }
        return samples.toArray(new double[0][][]);
    }

    private double[] createBoard(int ownMask, int enemyMask) {
        double[] board = new double[BOARD_SIZE];
        for (int position = 0; position < BOARD_SIZE; position++) {
            int bit = 1 << position;
            if ((ownMask & bit) != 0) {
                board[position] = OWN_PIECE;
            } else if ((enemyMask & bit) != 0) {
                board[position] = ENEMY_PIECE;
            } else {
                board[position] = EMPTY;
            }
        }
        return board;
    }

    private double[] chooseMove(double[] board) {
        List<int[]> legalMoves = getLegalMoves(board);
        if (legalMoves.isEmpty()) return new double[BOARD_SIZE];

        for (int[] move : legalMoves) {
            double[] nextBoard = applyMove(board, move);
            if (hasWinner(nextBoard, OWN_PIECE)) return encodeMove(move);
        }

        List<int[]> safeMoves = new ArrayList<>();
        for (int[] move : legalMoves) {
            double[] nextBoard = applyMove(board, move);
            if (!hasWinner(nextBoard, ENEMY_PIECE) && !hasImmediateWinningMove(nextBoard, ENEMY_PIECE)) {
                safeMoves.add(move);
            }
        }
        if (!safeMoves.isEmpty()) {
            int randomIndex = ThreadLocalRandom.current().nextInt(safeMoves.size());
            return encodeMove(safeMoves.get(randomIndex));
        }

        for (int[] move : legalMoves) {
            double[] nextBoard = applyMove(board, move);
            if (!hasImmediateWinningMove(nextBoard, ENEMY_PIECE)) return encodeMove(move);
        }

        return encodeMove(legalMoves.get(0));
    }

    private List<int[]> getLegalMoves(double[] board) {
        List<int[]> moves = new ArrayList<>();
        int ownPieces = count(board, OWN_PIECE);
        if (ownPieces < MAX_PIECES) {
            for (int destination = 0; destination < BOARD_SIZE; destination++) {
                if (board[destination] == EMPTY) moves.add(new int[] {-1, destination});
            }
        } else {
            for (int source = 0; source < BOARD_SIZE; source++) {
                if (board[source] != OWN_PIECE) continue;
                for (int destination = 0; destination < BOARD_SIZE; destination++) {
                    if (board[destination] == EMPTY) moves.add(new int[] {source, destination});
                }
            }
        }
        return moves;
    }

    private double[] applyMove(double[] board, int[] move) {
        double[] nextBoard = board.clone();
        if (move[0] >= 0) nextBoard[move[0]] = EMPTY;
        nextBoard[move[1]] = OWN_PIECE;
        return nextBoard;
    }

    private double[] encodeMove(int[] move) {
        double[] output = new double[BOARD_SIZE];
        output[move[1]] = WIN;
        if (move[0] >= 0) output[move[0]] = MOVE_FROM;
        return output;
    }

    private boolean hasImmediateWinningMove(double[] board, double player) {
        for (int position = 0; position < BOARD_SIZE; position++) {
            if (board[position] != EMPTY) continue;
            double[] nextBoard = board.clone();
            nextBoard[position] = player;
            if (hasWinner(nextBoard, player)) return true;
        }
        return false;
    }

    private boolean hasWinner(double[] board, double player) {
        for (int[] line : WINNING_LINES) {
            if (board[line[0]] == player && board[line[1]] == player && board[line[2]] == player) return true;
        }
        return false;
    }

    private int count(double[] board, double value) {
        int result = 0;
        for (double position : board) {
            if (position == value) result++;
        }
        return result;
    }
}
