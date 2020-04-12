# algorithm 1: feature-sign search
note:

$$ 
	y - Ax = 
	\left[ y_j - \sum_{k \in \N_p} A_{j,k} \cdot x_k \right]_{j \in \N_m} 
$$

so, $y - Ax$ is linear wrt $x_i$, so (assuming $L_2$ norm):
$$ 
	\frac{\partial \Vert y - Ax \Vert^2 }{ \partial x_i }
	= \frac{\partial}{ \partial x_i }
	\Bigg( 
		\sum\limits_{j \in \N_m} \left[ y_j - \sum_{k \in \N_p} A_{j,k} x_k \right]^2 
	\Bigg)
$$ 
$$
	= 2 \cdot \sum\limits_{j \in \N_m} 
		\left[ y_j - \sum_{k \in \N_p} A_{j,k} x_k \right] 
		\cdot \frac{\partial}{ \partial x_i } \left[ y_j - \sum_{k \in \N_p} A_{j,k} x_k \right]
$$
$$
	= 2 \cdot \sum\limits_{j \in \N_m} 
		\left[ y_j - \sum_{k \in \N_p} A_{j,k} x_k \right] 
		\cdot \left[ - A_{j,i} \right]
$$
Note that the term
$$
= 2 \left[ y_j - \sum_{k \in \N_p} A_{j,k} x_k \right] 
$$
will cancel out when we take the argmax over all $i$. This works assuming whatever value we pick to take the derivative at $x_i$ at is also the value we plugin in when we evaluate the derivative at soem $x_k$ where $i \neq k$. So, we are left with
$$
	= - \sum\limits_{j \in \N_m} A_{j,i}
$$
meaning that 
$$
	\argmax_i \left\vert \frac{\partial \Vert y - Ax \Vert^2 }{ \partial x_i } \right\vert
	= \argmax_i \left\vert - \sum\limits_{j \in \N_m} A_{j,i} \right\vert
	= \argmax_i \left\vert \sum\limits_{j \in \N_m} A_{j,i} \right\vert
$$


