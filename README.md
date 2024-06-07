# Fisher's linear discriminant for binary classification

Fisher's criterion set a linear combination that maximize betweenclass and minimize within class  variance by maximizing this expression:

$J(\textbf{w}) = \frac{\textbf{w}^T \textbf{S}_B \textbf{w}}{\textbf{w}^T \textbf{S}_w \textbf{w}}$,

where $S_B$ and $S_w$ are the between-class and within-class covariance matrices of input data. For the two-class problem, the weight vector is calculated as: 

$\textbf{W} = S^{-1}_{\textbf{W}} (\textbf{m}_m -\textbf{m}_b)$.

Where $\textbf{W}$ is a optimal vector that project the two classes maximizing betweenclass and minimizing within class variance and  $S^{−1}_W$ is the inverse of S_W.

Images below shows an optimal vector that was gotten by fisher's criteron and a non optimal vector:
![FC](https://github.com/aleperalesg/Fisher-s-criterion/assets/120703609/6aa72956-b1c5-4363-9236-0c8cf68a2645)
