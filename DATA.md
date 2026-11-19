# 资料字典

CSV 为 UTF-8，第一行为列名。日期为北京时间，时间带偏移；资料可见时刻与业务发生时刻分别以字段含义为准。空值不能默认做零。

### `appointments.csv`

趟次由预约标识区分，`vehicle` 只是车牌，不能跨趟归并。门许可用分号分隔。

字段：`trip`、`vehicle`、`requested_gate`、`available_gates`、`vehicle_class`、`promised_complete_by`。

### `signals.csv`

同一趟重复候车信号常见。`source_clock` 需要先按来源偏移还原；`arrived_at` 决定当时可用性。过磅和卸完是不同事件。

字段：`event`、`trip`、`vehicle`、`source`、`kind`、`gate`、`source_clock`、`arrived_at`。

### `capacity.csv`

时间为每个服务位一车的占用分钟。共享称重和跨门通道各是一个容量位；非共享位一次处理一车。机动线复用各门场地，与常驻位并行卸料。

字段：`station`、`kind`、`gate`、`small_min`、`large_min`、`shared_with`。

### `moves.csv`

调度时要继承已经确认的调令，建议尚未占用机动线。

字段：`notice`、`line`、`from_gate`、`to_gate`、`start_at`、`ready_at`、`state`。

### `policy.json`

时钟偏移为来源时钟减统一时钟。故障区间按统一时钟，无可用候车记录不代表没有排队；计划在当班结束前生效。预警关注整趟超过指定等待或预约完成时刻的风险。
