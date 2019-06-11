#include "paddle/fluid/operators/prod_op.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace paddle {
namespace operators {

using framework::OpKernelType;
using framework::Tensor;

class ProdOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) of ProdOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) of ProdOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ProdOp should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    int x_num_col_dims = ctx->Attrs().Get<int>("x_num_col_dims");
    int y_num_col_dims = ctx->Attrs().Get<int>("y_num_col_dims");

    VLOG(3) << "prod operator x.shape=" << x_dims << " y.shape=" << y_dims
            << " x_num_col_dims=" << x_num_col_dims
            << " y_num_col_dims=" << y_num_col_dims;

    PADDLE_ENFORCE_GT(
        x_dims.size(), x_num_col_dims,
        "The input tensor X's rank of ProdOp should be larger than "
        "x_num_col_dims.");
    PADDLE_ENFORCE_GT(
        y_dims.size(), y_num_col_dims,
        "The input tensor Y's rank of ProdOp should be larger than "
        "y_num_col_dims: %ld vs %ld",
        y_dims.size(), y_num_col_dims);

    auto x_mat_dims = framework::flatten_to_2d(x_dims, x_num_col_dims);
    auto y_mat_dims = framework::flatten_to_2d(y_dims, y_num_col_dims);

    PADDLE_ENFORCE_EQ(x_mat_dims[1], y_mat_dims[0],
                      "First matrix's width must be equal with second matrix's "
                      "height. %s, %s",
                      x_mat_dims[1], y_mat_dims[0]);
    std::vector<int64_t> output_dims;
    output_dims.reserve(
        static_cast<size_t>(x_num_col_dims + y_dims.size() - y_num_col_dims));

    for (int i = 0; i < x_num_col_dims; ++i) {
      output_dims.push_back(x_dims[i]);
    }

    for (int i = y_num_col_dims; i < y_dims.size(); ++i) {
      output_dims.push_back(y_dims[i]);
    }

    ctx->SetOutputDim("Out", framework::make_ddim(output_dims));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class ProdOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The first input tensor of prod op.");
    AddInput("Y", "(Tensor), The second input tensor of prod op.");
    AddOutput("Out", "(Tensor), The output tensor of prod op.");
    AddAttr<int>(
        "x_num_col_dims",
        R"DOC((int, default 1), The prod_op can take tensors with more than two
              dimensions as its inputs. If the input <span class="markdown-equation" id="equation-1"></span> is a tensor with more
             than two dimensions, <span class="markdown-equation" id="equation-1"></span> will be flattened into a two-dimensional
              matrix first. The flattening rule is: the first `num_col_dims`
              will be flattened to form the first dimension of the final matrix
              (the height of the matrix), and the rest `rank(X) - num_col_dims`
              dimensions are flattened to form the second dimension of the final
              matrix (the width of the matrix). As a result, height of the
              flattened matrix is equal to the product of <span class="markdown-equation" id="equation-1"></span>'s first
              `x_num_col_dims` dimensions' sizes, and width of the flattened
              matrix is equal to the product of <span class="markdown-equation" id="equation-1"></span>'s last `rank(x) - num_col_dims`
              dimensions' size. For example, suppose <span class="markdown-equation" id="equation-1"></span> is a 6-dimensional
              tensor with the shape [2, 3, 4, 5, 6], and `x_num_col_dims` = 3.
              Thus, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] =
              [24, 30].
        )DOC")
        .SetDefault(1)
        .EqualGreaterThan(1);
    AddAttr<int>(
        "y_num_col_dims",
        R"DOC((int, default 1), The prod_op can take tensors with more than two,
              dimensions as its inputs. If the input <span class="markdown-equation" id="equation-6"></span> is a tensor with more
              than two dimensions, <span class="markdown-equation" id="equation-6"></span> will be flattened into a two-dimensional
              matrix first. The attribute `y_num_col_dims` determines how <span class="markdown-equation" id="equation-6"></span> is
              flattened. See comments of `x_num_col_dims` for more details.
        )DOC")
        .SetDefault(1)
        .EqualGreaterThan(1);
    AddComment(R"DOC(
Prod Operator.

This operator is used to perform element-wise multiplication for input <span class="markdown-equation" id="equation-1"></span> and <span class="markdown-equation" id="equation-6"></span>.

The equation is:

$<span class="markdown-equation" id="equation-0"></span>$
)DOC");
  }
};

class ProdOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string> GetInputOutputWithSameType()
      const override {
    return std::unordered_map<std::string, std::string>{{"X", /*->*/ "Out"}};
  }
};

class ProdGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, y_dims);
    }
  }
};

class ProdOpGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> retv(new framework::OpDesc());
    retv->SetType("prod_grad");
    retv->SetInput("X", Input("X"));
    retv->SetInput("Y", Input("Y"));
    retv->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Y"), InputGrad("Y"));
    retv->SetAttrMap(Attrs());
    return retv;
  }
};

class ProdDoubleGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("DOut"), "Input(DOut) should not be null");

    if (ctx->HasOutput("DX")) {
      ctx->ShareDim("X", "DX");
    }
    if (ctx->HasOutput("DY")) {
      ctx->ShareDim("Y", "DY");
    }
    if (ctx->HasOutput("DDOut")) {
      ctx->ShareDim("DOut", "DDOut");
    }
  }
};

class ProdDoubleGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> retv(new framework::OpDesc());
    retv->SetType("prod_grad_grad");

    retv->SetInput("X", Input("X"));
    retv->SetInput("Y", Input("Y"));
    retv->SetInput("DOut", Input(framework::GradVarName("Out")));
    retv->SetInput("DDX", OutputGrad(framework::GradVarName("X")));
    retv->SetInput("DDY", OutputGrad(framework::GradVarName("Y")));

    retv->SetOutput("DDOut", InputGrad(framework::GradVarName("Out")));
    retv->SetOutput("DX", InputGrad("X"));
    retv->SetOutput("DY", InputGrad("Y"));

    retv->SetAttrMap(Attrs());
    return retv;
  }
};

} // namespace operators
} // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(prod, ops::ProdOp, ops::ProdOpMaker, ops::ProdOpInferVarType,
                  ops::ProdOpGradMaker);
REGISTER_OPERATOR(prod_grad, ops::ProdGradOp, ops::ProdDoubleGradMaker);
REGISTER_OPERATOR(prod_grad_grad, ops::ProdDoubleGradOp);
REGISTER_OP_CPU_KERNEL(
    prod, ops::ProdKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ProdKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    prod_grad, ops::ProdGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ProdGradKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    prod_grad_grad,
    ops::ProdDoubleGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ProdDoubleGradKernel<paddle::platform::CPUDeviceContext, double>);
