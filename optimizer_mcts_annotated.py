"""
AFlow蒙特卡洛树搜索优化流程详解

这段代码实现了AFlow论文中的核心优化算法，类似于AlphaGo中的MCTS，
但应用于自动化工作流的生成和优化。
"""

async def _optimize_graph(self):
    """
    🚀 AFlow的核心优化循环：蒙特卡洛树搜索工作流优化
    
    整体流程：
    1. 初始化：创建round_1的baseline
    2. 迭代优化：不断生成新的工作流变体
    3. 评估验证：测试新工作流的性能
    4. 更新经验：将成功的策略记录下来
    """
    validation_n = self.validation_rounds  # validation datasets's execution number
    graph_path = f"{self.root_path}/workflows"
    data = self.data_utils.load_results(graph_path)

    if self.round == 1:
        """
        🔄 第一轮：建立baseline
        - 使用最简单的工作流（通常是空prompt）
        - 获得初始性能分数作为后续优化的参考点
        """
        #NOTE(sjh) 存round的地址: round_1, round_2...
        directory = self.graph_utils.create_round_directory(graph_path, self.round) 

        # Load graph using graph_utils
        self.graph = self.graph_utils.load_graph(self.round, graph_path)
        # NOTE(sjh)这里的self.graph是Workflow类，不是Workflow对象

        avg_score = await self.evaluation_utils.evaluate_graph(self, directory, validation_n, data, initial=True)

    # 🔄 核心优化循环：蒙特卡洛树搜索
    while True:
        """
        📊 MCTS的4个核心步骤：Selection → Expansion → Simulation → Backpropagation
        
        在AFlow中对应：
        - Selection: 选择父节点轮次（高分轮次更容易被选中）
        - Expansion: 基于父节点生成新的工作流变体
        - Simulation: 评估新工作流在验证集上的性能
        - Backpropagation: 更新experience，指导后续优化
        """
        
        # 🎯 步骤1：Selection（选择阶段）
        # 创建新轮次的目录
        directory = self.graph_utils.create_round_directory(graph_path, self.round + 1)

        # 🔍 获取候选父节点：历史上表现最好的轮次
        top_rounds = self.data_utils.get_top_rounds(self.sample)
        """
        top_rounds示例：
        [
            {"round": 5, "score": 0.92},  # 最佳轮次
            {"round": 3, "score": 0.88},  # 次佳轮次  
            {"round": 1, "score": 0.80},  # baseline
            ...
        ]
        """
        
        # 🎲 使用软混合概率策略选择父节点
        sample = self.data_utils.select_round(top_rounds)
        """
        软混合概率策略的核心思想：
        - 高分轮次有更高概率被选中（利用exploitation）
        - 所有轮次都有被选中的机会（探索exploration）
        - 通过λ参数控制探索vs利用的平衡
        
        结果：sample = {"round": 5, "score": 0.92}
        """

        # 🧬 步骤2：Expansion（扩展阶段）
        # 读取被选中父节点的工作流代码和prompt
        prompt, graph_load = self.graph_utils.read_graph_files(sample["round"], graph_path)
        graph = self.graph_utils.extract_solve_graph(graph_load)
        """
        读取父节点的具体实现：
        - prompt: 父节点使用的指令模板
        - graph_load: 父节点的完整工作流代码
        - graph: 提取的核心逻辑部分
        """

        # 📚 收集优化所需的上下文信息
        processed_experience = self.experience_utils.load_experience()
        experience = self.experience_utils.format_experience(processed_experience, sample["round"])
        """
        Experience包含：
        - 历史上成功/失败的修改策略
        - 哪些操作符组合效果好
        - 避免重复已知的错误方向
        """
        
        operator_description = self.graph_utils.load_operators_description(self.operators)
        log_data = self.data_utils.load_log(sample["round"])
        """
        Additional Context:
        - operator_description: 可用操作符的说明文档
        - log_data: 父节点的实际执行日志（成功/失败案例）
        """

        # 🎨 构造优化prompt：指导LLM生成新的工作流
        graph_optimize_prompt = self.graph_utils.create_graph_optimize_prompt(
            experience, sample["score"], graph[0], prompt, operator_description, self.type, log_data
        )
        """
        优化prompt包含：
        - 父节点的当前实现和性能
        - 历史经验教训
        - 可用的操作符工具箱
        - 具体的改进目标和约束
        """

        # 🤖 调用优化LLM生成新的工作流
        try:
            # Create XmlFormatter based on GraphOptimize model
            graph_formatter = XmlFormatter.from_model(GraphOptimize)
            
            # Call the LLM with formatter
            response = await self.optimize_llm.call_with_format(
                graph_optimize_prompt, 
                graph_formatter
            )
            """
            LLM输出的新工作流包含：
            - modification: 具体的修改描述
            - new_graph: 新的工作流代码
            - new_prompt: 新的指令模板
            - reasoning: 修改的理由和预期效果
            """
            
            # If we reach here, response is properly formatted and validated
            logger.info(f"Graph optimization response received successfully")
        except FormatError as e:
            # Handle format validation errors
            logger.error(f"Format error in graph optimization: {str(e)}")
            # Try again with a fallback approach - direct call with post-processing
            raw_response = await self.optimize_llm(graph_optimize_prompt)
            
            # Try to extract fields using basic parsing
            response = self._extract_fields_from_response(raw_response)
            if not response:
                logger.error("Failed to extract fields from raw response, retrying...")
                continue

        # 🔍 步骤3：Simulation（模拟评估阶段）
        # 检查新生成的工作流是否符合约束条件
        check = self.experience_utils.check_modification(
            processed_experience, response["modification"], sample["round"]
        )
        """
        检查包括：
        - 是否重复了已知失败的修改
        - 是否符合基本的语法和逻辑要求
        - 是否在合理的修改范围内
        
        check = True: 新工作流通过初步验证，可以进入实际测试
        check = False: 新工作流有问题，需要重新生成
        """

        # 🎯 步骤4：Backpropagation（反向传播阶段）
        # If `check` is True, break the loop; otherwise, regenerate the graph
        if check:
            """
            ✅ 新工作流通过验证，结束当前轮次的生成
            
            接下来会：
            1. 将新工作流保存到round_N+1目录
            2. 在验证集上评估其性能
            3. 将结果和经验更新到experience数据库
            4. 为下一轮优化做准备
            
            这形成了完整的MCTS循环：
            好的修改 → 更高概率被选为父节点 → 产生更多好的后代
            坏的修改 → 较低概率被选中 → 逐渐被淘汰
            """
            break
        else:
            """
            ❌ 新工作流验证失败，重新开始MCTS循环
            
            系统会：
            1. 重新选择父节点（可能选中不同的轮次）
            2. 生成新的工作流变体
            3. 直到找到合适的候选方案
            
            这体现了MCTS的探索性：不会困在局部最优解
            """
            logger.info("Generated workflow failed validation, retrying with different approach...")
            continue 