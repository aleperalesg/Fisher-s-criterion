# Fisher's linear discriminant for binary classification

Fisher's criterion set a linear combination that maximize betweenclass and minimize within class  variance by maximizing this expression:

$J(\textbf{w}) = \frac{\textbf{w}^T \textbf{S}_B \textbf{w}}{\textbf{w}^T \textbf{S}_w \textbf{w}}$,

where $S_B$ and $S_w$ are the between-class and within-class covariance matrices of input data. For the two-class problem, the weight vector is calculated as: 

$\textbf{w} = S^{-1}_{\textbf{W}} (\textbf{m}_m -\textbf{m}_b)$


