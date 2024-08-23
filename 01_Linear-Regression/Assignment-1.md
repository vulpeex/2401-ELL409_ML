## Assignment-1 description 

Given a dataset of 10000 points (data.csv). 
Remove any outliers, if required. 
Build a Linear Regression model to fit the data where you have to minimize the Least Squares loss. 

The least squares loss is defined as:

$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right)^2$

##### Gradient Descent 
Write the algorithm of Gradient Descent from scratch. The goal is to minimize the cost function $J(\theta)$.

The update rule for Gradient Descent is given by

$\theta := \theta - \alpha \frac{\partial{J(\theta)}}{\partial{\theta}}$

where $\alpha$ is the learning rate. 

- Convergence criteria: The algorithm converges when the change in cost function $J(\theta)$ is less than a small value $\epsilon$, i.e.,
  
$|J(\theta^{t})-J(\theta^{t-1})|<\epsilon$

where, $\theta^{t}$ and $\theta^{t-1}$ are parameters at iterations t and t-1 respectively.

##### Stochastic Gradient Descent(SGD)
write the algorithm for Stochastic Gradient Descent from scratch. 
Unlike batch gradient descent, which uses the entire dataset, SGD updates the parameters for each training example $(x^i,y^i)$

$\theta := \theta - \alpha(h_{\theta}(x^i)-y^i)x^i$

-Convergence Criteria: The algorithm converges when the change in cost function $J(\theta)$ is less than a small value $\epsilon$, i.e.,

 $\frac{1}{k} \sum_{i=1}^{k} \left| J\left(\theta^{(t-i)}\right) - J\left(\theta^{(t-i-1)}\right) \right| < \epsilon$

where, k is the number of past interactions considered in the moving average. 

#### Submission Guidelines
1. Your submission should include the Python script named run.py.
2. The script takes following arguments as input:
    - data_path : Path to dataset. 
    - num_epochs: Number of epochs to run the algorithm.
    - batch_size: Batch size for Stochastic Gradient Descent. 
    - learning_rate: Learning rate for Gradient Descent.
3. Command to run your script should be in the format:
    ```
    python run.py-data_path <data_path>-num_epochs<num_epochs>-batch_size<batch_size>-learning_rate<learning_rate>
    ```
4. Output should be the final parameters learned by the linear regression model. 
5. Include a report (not more than two pages) detailing:
- Keep convergence criteria fixed at $\epsilon = 1e-5$
- The number of iterations it took to converge for varying learning rates $(lr=0.1, lr=0.01, lr=0.001)$.
-  The time for convergence for varying batch sizes(1, 10, 100, 1000).
- Training and validation loss plots.
