defmodule Scholar.SVM.SVM do
  @moduledoc """
  Support Vector Machine algorithm
  """
  import Nx.Defn
  import Scholar.Shared

  @derive {Nx.Container,
           keep: [:kernel, :mode, :gamma, :coef0, :degree],
           containers: [:coefficients, :bias, :data]}
  defstruct [:coefficients, :bias, :data, :kernel, :mode, :gamma, :coef0, :degree]

  opts = [
    num_classes: [
      required: true,
      type: :pos_integer,
      doc: "number of classes contained in the input tensors."
    ],
    learning_rate: [
      type: {:custom, Scholar.Options, :positive_number, []},
      default: 1.0e-3,
      doc: """
      learning rate used by gradient descent.
      """
    ],
    iterations: [
      type: :pos_integer,
      default: 1000,
      doc: """
      number of iterations of gradient descent performed inside logistic
      regression.
      """
    ],
    tol: [
      type: {:custom, Scholar.Options, :positive_number, []},
      default: 1.0e-4,
      doc: """
      Tolerance for stopping criteria.
      """
    ],
    key: [
      type: {:custom, Scholar.Options, :key, []},
      doc: """
      Determines random number generation for centroid initialization.
      If the key is not provided, it is set to `Nx.Random.key(System.system_time())`.
      """
    ],
    c: [
      type: {:custom, Scholar.Options, :positive_number, []},
      default: 1.0,
      doc: """
      Regularization parameter. The strength of the regularization is inversely proportional to C.
      Must be strictly positive.
      """
    ],
    gamma: [
      type: {:or, [{:in, [:scale, :auto]}, {:custom, Scholar.Options, :positive_number, []}]},
      default: :scale,
      doc: """
      Kernel coefficient for `rbf_kernel`, `poly_kernel` and `sigmoid_kernel`.
      """
    ],
    degree: [
      type: :pos_integer,
      default: 3,
      doc: """
      Degree of the polynomial kernel function (`polynomial_kernel`).
      Ignored by all other kernels.
      """
    ],
    coef0: [
      type: :float,
      default: 0.0,
      doc: """
      Independent term in kernel function. It is only significant in
      `polynomial_kernel` and `sigmoid_kernel`.
      """
    ],
    kernel: [
      type: {:in, [:linear_kernel, :polynomial_kernel, :rbf_kernel, :sigmoid_kernel]},
      default: :rbf_kernel,
      doc: """
      Specifies the kernel type to be used in the algorithm.
      It must be one of `:linear_kernel`, `:polynomial_kernel`, `:rbf_kernel` or `:sigmoid_kernel`.
      If none is given, `:rbf_kernel` will be used.
      """
    ]
  ]

  @opts_schema NimbleOptions.new!(opts)

  @doc """
  Fits a svm model for sample inputs `x` and sample
  targets `y`.


  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

    The function returns a struct with the following parameters:

    * `:coefficients` - Coefficient of the features in the decision function.

    * `:bias` - Bias added to the decision function.

    * `:coef0` - Independent term in kernel function. It is only significant in
      `polynomial_kernel` and `sigmoid_kernel`.

    * `:degree` - Degree of the polynomial kernel function (`polynomial_kernel`).

    * `:gamma` - Kernel coefficient for `rbf_kernel`, `poly_kernel` and `sigmoid_kernel`.

    * `:kernel` - Specifies the kernel type to be used in the algorithm.

    * `:mode` - Specifies the mode of the kernel.

    * `:data` - Training data.

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([1, 0, 1])
      iex> Scholar.SVM.SVM.fit(x, y, num_classes: 2)
      %Scholar.SVM.SVM{
        coefficients: #Nx.Tensor<
          f32[2][3]
          [
            [-0.21080760657787323, 0.2100450098514557, -0.21076801419258118],
            [0.21080760657787323, -0.2100450098514557, 0.21076801419258118]
          ]
        >,
        bias: #Nx.Tensor<
          f32[2]
          [-0.21076801419258118, 0.21076801419258118]
        >,
        data: #Nx.Tensor<
          f32[3][2]
          [
            [1.0, 2.0],
            [3.0, 2.0],
            [4.0, 7.0]
          ]
        >,
        kernel: :rbf_kernel,
        mode: :scale,
        gamma: :scale,
        coef0: 0.0,
        degree: 3
      }
  """
  deftransform fit(x, targets, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    mode =
      case opts[:gamma] do
        :scale -> :scale
        :auto -> :auto
        _ -> :custom
      end

    opts = Keyword.put(opts, :mode, mode)
    fit_n(x, targets, opts)
  end

  defn fit_n(x, targets, opts \\ []) do
    tol = opts[:tol]
    iterations = opts[:iterations]
    learning_rate = opts[:learning_rate]
    num_classes = opts[:num_classes]
    c = opts[:c]
    coef0 = opts[:coef0]
    degree = opts[:degree]

    {num_samples, _num_features} = Nx.shape(x)

    params = Nx.broadcast(Nx.tensor(0.0, type: to_float_type(x)), {num_classes, num_samples})
    bias = Nx.broadcast(Nx.tensor(0.0, type: to_float_type(x)), {num_classes})
    prev_params = params + 10 * tol

    # Training loop
    {final_params, final_bias, _, _, _, _, _, _, _} =
      while {params, bias, prev_params, x, targets, learning_rate, tol,
             binary_targets_all = Nx.select(targets == Nx.iota({num_classes, 1}), 1.0, -1.0),
             i = 0},
            i < iterations and Nx.all(Nx.abs(params - prev_params)) > tol do
        {params, bias, _, _, _, _, _, _, _, _} =
          while {params, bias, prev_params, x, targets, learning_rate, tol, binary_targets_all, i,
                 j = 0},
                j < num_classes do
            binary_targets = Nx.take(binary_targets_all, j, axis: 0)

            prev_params =
              Nx.put_slice(prev_params, [j, 0], Nx.new_axis(Nx.take(params, j, axis: 0), 0))

            {grad_params, grad_bias} =
              grad_svm_loss(
                Nx.take(params, j, axis: 0),
                Nx.take(bias, j),
                x,
                binary_targets,
                c,
                coef0,
                mode: opts[:mode],
                gamma: opts[:gamma],
                kernel: opts[:kernel],
                degree: opts[:degree]
              )

            params =
              Nx.put_slice(
                params,
                [j, 0],
                Nx.new_axis(Nx.take(params, j, axis: 0) - learning_rate * grad_params, 0)
              )

            bias =
              Nx.put_slice(
                bias,
                [j],
                Nx.new_axis(Nx.take(bias, j) - learning_rate * grad_bias, 0)
              )

            {params, bias, prev_params, x, targets, learning_rate, tol, binary_targets_all, i,
             j + 1}
          end

        {params, bias, prev_params, x, targets, learning_rate, tol, binary_targets_all, i + 1}
      end

    %__MODULE__{
      coefficients: final_params,
      bias: final_bias,
      data: x,
      coef0: coef0,
      kernel: opts[:kernel],
      mode: opts[:mode],
      gamma: opts[:gamma],
      degree: degree
    }
  end

  defnp rbf_kernel(x, y, opts \\ []) do
    mode = opts[:mode]
    {num_features} = Nx.shape(x)

    gamma =
      case mode do
        :scale ->
          1 / (num_features * Nx.variance(x))

        :auto ->
          1 / num_features

        _ ->
          opts[:gamma]
      end

    Nx.exp(-gamma * Nx.sum((x - y) ** 2))
  end

  defnp linear_kernel(x, y) do
    Nx.dot(x, y)
  end

  defnp polynomial_kernel(x, y, coef0, degree, opts \\ []) do
    mode = opts[:mode]
    {num_features} = Nx.shape(x)

    gamma =
      case mode do
        :scale ->
          1 / (num_features * Nx.variance(x))

        :auto ->
          1 / num_features

        _ ->
          opts[:gamma]
      end

    (coef0 + gamma * Nx.dot(x, y)) ** degree
  end

  defnp sigmoid_kernel(x, y, coef0, opts \\ []) do
    mode = opts[:mode]
    {num_features} = Nx.shape(x)

    gamma =
      case mode do
        :scale ->
          1 / (num_features * Nx.variance(x))

        :auto ->
          1 / num_features

        _ ->
          opts[:gamma]
      end

    Nx.tanh(coef0 + gamma * Nx.dot(x, y))
  end

  defn svm_loss(coefficients, bias, x, targets, c, coef0, opts \\ []) do
    xi = Nx.vectorize(x, :i)
    xj = Nx.vectorize(x, :j)

    inner_sum =
      case opts[:kernel] do
        :rbf_kernel ->
          rbf_kernel(xi, xj, mode: opts[:mode], gamma: opts[:gamma])

        :linear_kernel ->
          linear_kernel(xi, xj)

        :polynomial_kernel ->
          polynomial_kernel(xi, xj, coef0, opts[:degree], mode: opts[:mode], gamma: opts[:gamma])

        :sigmoid_kernel ->
          sigmoid_kernel(xi, xj, coef0, mode: opts[:mode], gamma: opts[:gamma])
      end
      |> Nx.devectorize()
      |> Nx.devectorize()

    final_predictions = Nx.dot(inner_sum, coefficients) + bias
    hinge_loss = Nx.max(0, 1 - final_predictions * targets)
    reg_term = 0.5 * (Nx.sum(coefficients ** 2) + Nx.sum(bias ** 2)) / c
    Nx.mean(hinge_loss) + reg_term
  end

  defn grad_svm_loss(coefficients, bias, x, targets, l2_reg, coef0, opts \\ []) do
    grad({coefficients, bias}, fn {coefficients, bias} ->
      svm_loss(coefficients, bias, x, targets, l2_reg, coef0, opts)
    end)
  end

  defn predict(
         %__MODULE__{
           coefficients: coefficients,
           bias: bias,
           data: data,
           mode: mode,
           kernel: kernel,
           gamma: gamma,
           coef0: coef0,
           degree: degree
         },
         x
       ) do
    xi = Nx.vectorize(x, :i)
    xj = Nx.vectorize(data, :j)

    inner_sum =
      case kernel do
        :rbf_kernel -> rbf_kernel(xi, xj, mode: mode, gamma: gamma)
        :linear_kernel -> linear_kernel(xi, xj)
        :polynomial_kernel -> polynomial_kernel(xi, xj, coef0, degree, mode: mode, gamma: gamma)
        :sigmoid_kernel -> sigmoid_kernel(xi, xj, coef0, mode: mode, gamma: gamma)
      end
      |> Nx.devectorize()
      |> Nx.devectorize()

    final_predictions = Nx.dot(inner_sum, [1], coefficients, [1]) + bias
    Nx.argmax(final_predictions, axis: 1)
  end

  defn predict_new(
         %__MODULE__{
           coefficients: coefficients,
           bias: bias,
           data: data,
           mode: mode,
           kernel: kernel,
           gamma: gamma,
           coef0: coef0,
           degree: degree
         },
         x
       ) do
    {num_samples, _} = Nx.shape(x)
    {num_classes, _} = Nx.shape(coefficients)
    xi = Nx.vectorize(x, :i)
    xj = Nx.vectorize(data, :j)

    inner_sum =
      case kernel do
        :rbf_kernel -> rbf_kernel(xi, xj, mode: mode, gamma: gamma)
        :linear_kernel -> linear_kernel(xi, xj)
        :polynomial_kernel -> polynomial_kernel(xi, xj, coef0, degree, mode: mode, gamma: gamma)
        :sigmoid_kernel -> sigmoid_kernel(xi, xj, coef0, mode: mode, gamma: gamma)
      end
      |> Nx.devectorize()
      |> Nx.devectorize()

    labels = Nx.broadcast(0, {num_samples})
    votes = Nx.broadcast(0.0, {num_samples, num_classes})

    {final_labels, final_votes, _, _, _, _, _, _} =
      while {labels, votes, inner_sum, coefficients, bias, num_samples, num_classes, k = 0},
            k < num_samples do
        {num_classes, _} = Nx.shape(coefficients)
        vote = Nx.broadcast(0.0, {num_classes})

        {final_vote, _, _, _, _, _, _} =
          while {vote, inner_sum, coefficients, bias, num_classes, k, i = 0}, i < num_classes do
            {vote, _, _, _, _, _, _, _} =
              while {vote, inner_sum, coefficients, bias, num_classes, k, i, j = i + 1},
                    j < num_classes do
                coef1 = Nx.take(coefficients, i, axis: 0)
                coef2 = Nx.take(coefficients, j, axis: 0)

                sum =
                  Nx.dot(Nx.take(inner_sum, k),  coef1) +
                    Nx.dot(Nx.take(inner_sum, k), coef2) -
                    Nx.take(bias, i) - Nx.take(bias, j)

                vote =
                  if sum > 0,
                    do: Nx.indexed_add(vote, Nx.new_axis(Nx.new_axis(i, -1), -1), Nx.tensor([1])),
                    else: Nx.indexed_add(vote, Nx.new_axis(Nx.new_axis(j, -1), -1), Nx.tensor([1]))

                {vote, inner_sum, coefficients, bias, num_classes, k, i, j + 1}
              end

            {vote, inner_sum, coefficients, bias, num_classes, k, i + 1}
          end

        votes = Nx.put_slice(votes, [k, 0], Nx.new_axis(final_vote, 0))
        labels = Nx.indexed_put(labels, Nx.new_axis(k, -1), Nx.argmax(final_vote))
        {labels, votes, inner_sum, coefficients, bias, num_samples, num_classes, k + 1}
      end

    {final_votes, final_labels}
  end
end
