# robust_transform

`robust_transform` 是一套基于 `srcML` 的 C 代码语义保持变换工具。它的目标不是简单做字符串替换，而是先把 C 代码转成 srcML XML，再在结构化语法树上做变换，最后还原回 C 源码，用于数据增强、代码鲁棒性实验和变换效果分析。

这套实现支持两类使用方式：

- 单独运行某一类变换，生成某一类别的数据增强结果
- 混合多个变换，生成综合增强数据集

仓库里的变换主要覆盖四个层面：

- 词法层：变量名、参数名等标识符改写
- 表达式层：条件表达式和字面量的等价重写
- 控制流层：循环和语句包裹等结构调整
- 数据流 / 封装层：字面量提取、声明拆分、helper 函数封装

## 1. 目录结构

```text
robust_transform/
├─ augment_c_code.py              # 混合变换入口
├─ lexical_transform.py           # 词法变换入口
├─ expression_transform.py        # 表达式变换入口
├─ control_flow_transform.py      # 控制流变换入口
├─ dataflow_encap_transform.py    # 数据流/封装变换入口
├─ srcml_transforms.py            # 所有 srcML 变换的核心实现
├─ transform_utils.py             # XML/srcML/数据集处理公共工具
├─ name_obfuscator.py             # 标识符混淆名生成器
├─ test_srcml_transforms.py       # 每种变换的最小回归测试
└─ README.md
```

## 2. 整体工作流程

每个变换的基本流程一致：

1. 读取原始 C 代码
2. 使用 `srcml` 将 C 代码转换为 XML
3. 在 XML 树上定位目标节点并做结构化修改
4. 将修改后的 XML 再转回 C 源码
5. 将变换结果写回数据集 JSON

对应的核心函数主要在 [transform_utils.py](./transform_utils.py)：

- `code_to_srcml_root`：源码转 srcML XML
- `srcml_root_to_code`：srcML XML 转回源码
- `apply_srcml_mutation`：统一执行一次 XML 变换
- `build_augmented_dataset`：基于原始数据集构造增强数据集
- `transform_code_by_pool`：从某个变换池里随机采样若干个变换并组合执行

## 3. 依赖环境

建议环境：

- Python 3.10+
- `srcml`
- `gcc` 或 `clang`

其中：

- `srcml` 是运行变换所必须的
- `gcc` 不是生成数据必须的，但建议安装，用于编译级回归测试

### 安装 srcML

Windows 下安装完成后，确保命令行可直接执行：

```powershell
srcml --version
```

### 检查 gcc

```powershell
gcc --version
```

## 4. 输入输出数据格式

这些脚本面向的数据集是一个 JSON 列表，每个元素至少需要包含：

```json
[
  {
    "id": 1,
    "code": "int add(int a, int b) { return a + b; }"
  }
]
```

增强后会保留原字段，并增加：

- `source_id`：原样本 id
- `variant_index`：第几个变体
- `is_transformed`：是否为变换后的样本
- `transform_category`：变换类别
- `transformations`：实际应用到该样本上的变换名列表

## 5. 入口脚本说明

### `lexical_transform.py`

词法级增强入口。只从 `LEXICAL_TRANSFORMS` 中选变换。

适合：

- 只研究命名扰动对模型的影响
- 只做“代码结构不动、名字变化”的增强实验

### `expression_transform.py`

表达式级增强入口。只从 `EXPRESSION_TRANSFORMS` 中选变换。

适合：

- 条件表达式扰动
- 常量和字面量等价重写

### `control_flow_transform.py`

控制流级增强入口。只从 `CONTROL_FLOW_TRANSFORMS` 中选变换。

适合：

- 循环结构变化
- 语句包装
- 死代码注入

### `dataflow_encap_transform.py`

数据流与封装级增强入口。只从 `DATAFLOW_TRANSFORMS` 中选变换。

适合：

- 字面量提取
- 声明/赋值拆分
- helper 函数封装

### `augment_c_code.py`

综合增强入口。会从 `ALL_TRANSFORMS` 中抽取若干个变换组合执行。

适合：

- 构造混合型增强数据集
- 做更加接近真实扰动分布的训练数据

## 6. 运行方式

建议在项目根目录运行，且尽量显式传入 `--input` 和 `--output`，不要完全依赖默认路径。

### 运行单一类别变换

```powershell
python robust_transform/lexical_transform.py `
  --input dataset_v1/BaseCodeFilesReason.json `
  --output dataset_v1_lexical/BaseCodeFilesReason_lexical.json `
  --variants-per-sample 1 `
  --max-transforms-per-variant 3 `
  --seed 42 `
  --drop-original
```

```powershell
python robust_transform/expression_transform.py `
  --input dataset_v1/BaseCodeFilesReason.json `
  --output dataset_v1_expression/BaseCodeFilesReason_expression.json `
  --variants-per-sample 1 `
  --max-transforms-per-variant 4 `
  --seed 42 `
  --drop-original
```

```powershell
python robust_transform/control_flow_transform.py `
  --input dataset_v1/BaseCodeFilesReason.json `
  --output dataset_v1_control/BaseCodeFilesReason_control.json `
  --variants-per-sample 1 `
  --max-transforms-per-variant 3 `
  --seed 42 `
  --drop-original
```

```powershell
python robust_transform/dataflow_encap_transform.py `
  --input dataset_v1/BaseCodeFilesReason.json `
  --output dataset_v1_dataflow/BaseCodeFilesReason_dataflow.json `
  --variants-per-sample 1 `
  --max-transforms-per-variant 3 `
  --seed 42 `
  --drop-original
```

### 运行综合混合变换

```powershell
python robust_transform/augment_c_code.py `
  --input dataset_v1/BaseCodeFilesReason.json `
  --output dataset_v1_augmented/BaseCodeFilesReason_augmented.json `
  --variants-per-sample 2 `
  --max-transforms-per-variant 3 `
  --seed 42 `
  --drop-original
```

### 只启用指定变换

例如只启用两个表达式变换：

```powershell
python robust_transform/expression_transform.py `
  --input dataset_v1/BaseCodeFilesReason.json `
  --output tmp_expression.json `
  --transforms swap_symmetric_comparisons wrap_numeric_literals_parentheses
```

## 7. 通用参数说明

所有入口脚本共享一套通用参数：

- `--input`：输入 JSON 数据集路径
- `--output`：输出 JSON 路径
- `--variants-per-sample`：每条样本生成多少个变体
- `--max-transforms-per-variant`：一次样本最多叠加多少个变换
- `--seed`：随机种子，保证可复现
- `--drop-original`：只输出变换样本，不保留原样本
- `--transforms`：手动指定可启用的变换集合

## 8. 各个核心文件是做什么的

### `transform_utils.py`

这是整个模块的基础设施层，主要负责：

- srcML XML namespace 管理
- XML 树父子关系、兄弟节点、祖先节点辅助函数
- 通用 XML 节点构造
  - `make_decl_stmt`
  - `make_if_stmt`
  - `make_while_stmt`
  - `make_call_node`
  - `make_return_stmt`
- srcML 与源码互转
- 数据集读取与保存
- 多变换组合调度

还提供了一个调试用环境变量：

```powershell
$env:ROBUST_TRANSFORM_STRICT="1"
```

默认情况下，某个样本在变换时如果抛异常，会回退到原代码；设置 `ROBUST_TRANSFORM_STRICT=1` 后，异常会直接抛出，便于定位变换 bug。

### `srcml_transforms.py`

这里是所有语义变换的核心实现文件，包含：

- 具体 mutator 函数
- 每类变换的注册字典
  - `LEXICAL_TRANSFORMS`
  - `EXPRESSION_TRANSFORMS`
  - `CONTROL_FLOW_TRANSFORMS`
  - `DATAFLOW_TRANSFORMS`
- 总注册表 `ALL_TRANSFORMS`

如果你要新增一种变换，通常需要：

1. 在这里实现一个 `code -> code` 的变换函数
2. 把它注册进相应的变换字典
3. 为它补一条测试用例

### `name_obfuscator.py`

用于生成新的变量名、参数名、临时名。当前实现支持多种“混淆风格”，例如：

- 随机长 token
- 看起来容易混淆的字符组合
- 带有伪指令/注入风格的字符串

它被词法变换和部分数据流变换复用。

### `test_srcml_transforms.py`

这是当前发布版本对应的最小回归测试。

测试内容包括：

- 每一种变换都能实际触发
- 变换结果包含预期结构
- 变换后的 C 代码可以通过 `gcc -fsyntax-only`

## 9. 13 种变换分别做什么

下面按类别说明每种变换的目的、做法和注意点。

### A. 词法层变换

#### `obfuscate_local_variables`

作用：

- 只混淆函数内部局部变量名

怎么做：

- 收集函数体中的局部变量声明
- 过滤关键字、保留名、全大写名、`main`
- 使用 `AdvancedNameObfuscator` 生成新名字
- 在安全引用位置统一替换名字

安全策略：

- 不改类型名
- 不改 `goto`/`label`
- 不改函数调用名
- 不改成员访问里的字段名，如 `obj.x`、`ptr->x`

#### `obfuscate_parameters`

作用：

- 只混淆函数参数名

怎么做：

- 定位 `parameter -> decl -> name`
- 用和局部变量同样的安全替换逻辑替换引用点

适用场景：

- 测试模型对参数命名扰动的鲁棒性

#### `obfuscate_local_identifiers`

作用：

- 同时混淆参数和局部变量

怎么做：

- 综合前两种逻辑，在同一函数内部挑选可重命名标识符进行替换

### B. 表达式层变换

#### `swap_symmetric_comparisons`

作用：

- 将对称比较重写为反向比较

示例：

```c
if (a == b)
```

变成：

```c
if (b == a)
```

怎么做：

- 定位 `==` 和 `!=`
- 取左右操作数节点
- 在保证不是复杂成员访问/下标等风险场景时交换两边

#### `append_identity_condition`

作用：

- 在条件表达式后追加恒等逻辑项

示例：

```c
if (x < y)
```

可能变成：

```c
if (x < y || 0)
if (x < y && 1)
```

怎么做：

- 找到 `if` 和 `while` 的条件表达式
- 排除包含赋值操作的条件
- 随机在尾部拼接 `|| 0` 或 `&& 1`

#### `wrap_numeric_literals_identity`

作用：

- 用恒等算术表达式包裹数字字面量

示例：

```c
7
```

可能变成：

```c
(7 + 0)
(7 - 0)
(7 * 1)
```

怎么做：

- 遍历数字字面量
- 随机选择一种恒等形式替换原字面量节点

#### `wrap_numeric_literals_parentheses`

作用：

- 给数字字面量额外加括号

示例：

```c
7
```

变成：

```c
(7)
```

怎么做：

- 定位数字字面量
- 排除本来就在括号里的情况
- 用 `<expr>(literal)</expr>` 替换原字面量

### C. 控制流层变换

#### `convert_for_to_while`

作用：

- 把 `for` 循环改写成等价的 `while` 循环

示例：

```c
for (init; cond; incr) {
    body;
}
```

变成：

```c
{
    init;
    while (cond) {
        body;
        incr;
    }
}
```

怎么做：

- 拆出 `init`、`condition`、`incr`
- 复制循环体
- 把 `incr` 追加到循环体末尾
- 用外层 block 包住初始化和 while

安全限制：

- 当前实现会跳过包含 `continue` 的 `for`
- 原因是 `continue` 在 `while` 版本里可能绕过追加的 `incr`，语义会变

#### `wrap_statements_if_true`

作用：

- 用恒真条件包裹原语句

示例：

```c
x++;
```

变成：

```c
if (1) {
    x++;
}
```

怎么做：

- 在 block 里随机挑选若干 `expr_stmt` / `return` / `break` / `continue`
- 用 `if (1)` 外包一层 block

#### `inject_dead_code`

作用：

- 注入不会执行的死代码分支

示例：

```c
if (0) {
    int dead_1 = 100;
    dead_1++;
}
```

怎么做：

- 在函数块中选定一个或多个插入位置
- 构造 `if (0)` 语句块
- 插入到 block 内部

用途：

- 增强样本表面复杂度
- 测试模型抗无效路径干扰能力

### D. 数据流 / 封装层变换

#### `extract_call_argument_literals`

作用：

- 把函数调用实参里的字面量提到前面的临时变量中

示例：

```c
printf("%d", 7);
```

变成：

```c
const char* t1 = "%d";
int t2 = 7;
printf(t1, t2);
```

怎么做：

- 定位 `call -> argument -> literal`
- 找到其所属的语句节点
- 在该语句前插入声明
- 用变量名替换原字面量

安全限制：

- 当前只在真实 block 中做前插声明
- 对 srcML 的 `pseudo block` 会跳过
- 这样可以避免把无花括号单语句 `for/if/while` 改成非法 C 代码

#### `split_declaration_and_initialization`

作用：

- 将变量声明与初始化拆成两条语句

示例：

```c
int x = 3;
```

变成：

```c
int x;
x = 3;
```

怎么做：

- 定位只有一个 `decl` 的 `decl_stmt`
- 删除其中的 `init`
- 在原声明后追加一条赋值语句

安全限制：

- 跳过 `const`、`static`、`auto`、引用等不适合拆分的声明
- 跳过 `init` / `condition` / 顶层全局声明等场景

#### `encapsulate_literals_with_helpers`

作用：

- 把函数内部字面量替换为 helper 函数调用

示例：

```c
return x + 7;
```

变成：

```c
int get_hardcode_1() { return 7; }
return x + get_hardcode_1();
```

怎么做：

- 遍历函数体内字面量
- 为每个字面量生成一个 helper 函数
- 用 `get_hardcode_n()` 调用替换原字面量
- 将 helper 函数插入到顶层定义区

安全策略：

- helper 名会避开现有全局函数/声明名
- 跳过 `case` 标签、数组下标等更敏感的位置

## 10. 语义保持是如何尽量保证的

这套实现追求的是“尽量语义保持”，不是形式化证明的绝对等价。主要靠以下手段降低风险：

- 基于语法树节点操作，而不是裸字符串替换
- 跳过容易破坏语义的场景
  - `continue` 的 `for -> while`
  - 成员访问、类型名、label、goto
  - 无花括号单语句体中的前插声明
- 对失败变换自动回退到原代码
- 提供最小化编译回归测试

## 11. 已知限制

发布时建议明确说明以下限制：

- 当前主要面向 C，不是完整 C++ 变换框架
- 变换结果的格式化风格可能和原代码不同
- 个别变换带随机性，同一输入在不同 seed 下结果可能不同
- “能编译”不等于“对所有复杂边界场景都严格语义等价”
- 部分入口脚本当前默认输入输出路径并不完全统一，发布时建议用户显式传参

## 12. 测试方法

项目已包含最小回归测试文件 [test_srcml_transforms.py](./test_srcml_transforms.py)。

运行方式：

```powershell
python -m unittest robust_transform\test_srcml_transforms.py -v
```

测试会检查：

- 每种变换都能实际触发
- 输出包含预期结构
- 输出代码可以通过 `gcc -fsyntax-only`

## 13. 如果你要新增一种变换

推荐流程：

1. 在 [srcml_transforms.py](./srcml_transforms.py) 中实现新的 mutator
2. 封装成 `def transform_name(code: str, rng: random.Random) -> str`
3. 注册到对应的变换字典
4. 在 [test_srcml_transforms.py](./test_srcml_transforms.py) 中增加一个最小测试样例
5. 跑一遍：

```powershell
python -m unittest robust_transform\test_srcml_transforms.py -v
```

## 14. 推荐引用方式

如果你准备把它发布到 GitHub，建议在仓库主页说明中突出下面几点：

- 这是一个 `srcML-based semantic-preserving C code augmentation toolkit`
- 支持 lexical / expression / control-flow / dataflow 四类变换
- 支持 JSON 数据集批量增强
- 包含最小可运行测试，便于复现实验

如果你愿意，我还可以继续帮你做两件事：

1. 把这份 README 再整理成更适合 GitHub 展示的“中英双语版”。  
2. 顺手补一个仓库根目录级别的英文 `README.md` 简介和示例截图结构。  
