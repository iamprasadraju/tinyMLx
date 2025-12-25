import numpy as np

"""

	A = [1, 0, 4, 3]    | 	B = [0, 1, 3]
		[3, 2, 1, 8]	|	    [3, 8, 9]
		[1, 9, 6, 0]	| 	    [0, 8, 3]
								[8, 6, 3]
		
	AB[0][0] = A[0][0] * B[0][0] + A[0][1] * A[1][0] + A[0][2] * A[2][0]
	
	- A[row, col] @ B[row, col] -> no.of elemennts in col of A = no.of elemennts row of B
	
	:) if A: [2, 3] B: [3, 2] --> AB:[2, 2] 

"""

def matmul(matrixA, matrixB):
	rowsA = matrixA.shape[0]
	colsA = matrixA.shape[1]
	
	rowsB = matrixB.shape[0]
	colsB = matrixB.shape[1]

	if colsA == rowsB:
		matrixC = np.zeros((rowsA, colsB))
		for i in range(rowsA):
			for j in range(colsB):
				acc = 0
				for k in range(colsA):
					acc += matrixA[i][k] * matrixB[k][j]
				matrixC[i][j] = acc
		return matrixC
			
	else:
		raise ValueError("no.of cols of A != no.of rows of B")


def npmatmul(matrixA, matrixB):
	return matrixA @ matrixB
	