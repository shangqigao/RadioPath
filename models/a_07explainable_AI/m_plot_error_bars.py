import plotly.graph_objects as go
import numpy as np

# # radiomics
# # plot c index
# fig = go.Figure()
# fig.add_trace(go.Bar(
#     name='Pyradiomics',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.588, 0.542, 0.527, 0.524, 0.545],
#     error_y=dict(type='data', array=[0.057, 0.085, 0.107, 0.058, 0.077])
# ))
# fig.add_trace(go.Bar(
#     name='M3D-CLIP',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.542, 0.584, 0.554, 0.593, 0.568],
#     error_y=dict(type='data', array=[0.056, 0.073, 0.079, 0.073, 0.070])
# ))
# fig.add_trace(go.Bar(
#     name='SegVol ViT + Mean',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.575, 0.593, 0.639, 0.591, 0.600],
#     error_y=dict(type='data', array=[0.053, 0.061, 0.077, 0.076, 0.067])
# ))
# fig.add_trace(go.Bar(
#     name='SegVol ViT + ABMIL',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.623, 0.610, 0.624, 0.614, 0.618],
#     error_y=dict(type='data', array=[0.036, 0.063, 0.034, 0.078, 0.053])
# ))
# fig.add_trace(go.Bar(
#     name='SegVol ViT + SPARRA',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.639, 0.644, 0.647, 0.643, 0.643],
#     error_y=dict(type='data', array=[0.084, 0.088, 0.071, 0.089, 0.083])
# ))
# fig.update_layout(
#     barmode='group',
#     xaxis=dict(
#         title=dict(
#             text='Survival model'
#         ),
#         title_font=dict(size=20),
#         tickangle=0,
#         tickfont=dict(
#         size=20
#         ),
#     ),
#     yaxis=dict(
#         title=dict(
#             text='C-Index'
#         ),
#         title_font=dict(size=20)
#     ),
#     margin=dict(t=10, l=50, r=30),
#     legend=dict(
#         x=0.01, y=0.01,
#         xanchor='left', yanchor='bottom',
#         font=dict(size=20)
#     )
# )
# fig.write_image(f"a_07explainable_AI/survival_metric_cindex.png", width=1000, height=500, scale=2)

# # plot c ipcw
# fig = go.Figure()
# fig.add_trace(go.Bar(
#     name='Pyradiomics',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.624, 0.459, 0.392, 0.452, 0.482],
#     error_y=dict(type='data', array=[0.178, 0.153, 0.092, 0.138, 0.140])
# ))
# fig.add_trace(go.Bar(
#     name='M3D-CLIP',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.667, 0.743, 0.743, 0.762, 0.729],
#     error_y=dict(type='data', array=[0.094, 0.059, 0.072, 0.058, 0.071])
# ))
# fig.add_trace(go.Bar(
#     name='SegVol ViT + Mean',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.575, 0.697, 0.749, 0.705, 0.682],
#     error_y=dict(type='data', array=[0.167, 0.100, 0.072, 0.097, 0.109])
# ))
# fig.add_trace(go.Bar(
#     name='SegVol ViT + ABMIL',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.679, 0.736, 0.754, 0.750, 0.730],
#     error_y=dict(type='data', array=[0.159, 0.062, 0.066, 0.032, 0.080])
# ))
# fig.add_trace(go.Bar(
#     name='SegVol ViT + SPARRA',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.574, 0.723, 0.755, 0.735, 0.670],
#     error_y=dict(type='data', array=[0.148, 0.097, 0.064, 0.088, 0.090])
# ))
# fig.update_layout(
#     barmode='group',
#     xaxis=dict(
#         title=dict(
#             text='Survival model'
#         ),
#         title_font=dict(size=20),
#         tickangle=0,
#         tickfont=dict(
#         size=20
#         ),
#     ),
#     yaxis=dict(
#         title=dict(
#             text='C-IPCW'
#         ),
#         title_font=dict(size=20)
#     ),
#     margin=dict(t=10, l=50, r=30),
#     legend=dict(
#         x=0.01, y=0.01,
#         xanchor='left', yanchor='bottom',
#         font=dict(size=20)
#     )
# )
# fig.write_image(f"a_07explainable_AI/survival_metric_cipcw.png", width=1000, height=500, scale=2)

# # plot c auc
# fig = go.Figure()
# fig.add_trace(go.Bar(
#     name='Pyradiomics',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.562, 0.521, 0.516, 0.501, 0.525],
#     error_y=dict(type='data', array=[0.067, 0.076, 0.081, 0.058, 0.071])
# ))
# fig.add_trace(go.Bar(
#     name='M3D-CLIP',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.550, 0.590, 0.566, 0.602, 0.577],
#     error_y=dict(type='data', array=[0.055, 0.089, 0.095, 0.094, 0.083])
# ))
# fig.add_trace(go.Bar(
#     name='SegVol ViT + Mean',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.584, 0.594, 0.637, 0.595, 0.603],
#     error_y=dict(type='data', array=[0.057, 0.059, 0.078, 0.070, 0.066])
# ))
# fig.add_trace(go.Bar(
#     name='SegVol ViT + ABMIL',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.616, 0.612, 0.610, 0.613, 0.613],
#     error_y=dict(type='data', array=[0.057, 0.079, 0.025, 0.096, 0.064])
# ))
# fig.add_trace(go.Bar(
#     name='SegVol ViT + SPARRA',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.647, 0.632, 0.659, 0.645, 0.646],
#     error_y=dict(type='data', array=[0.090, 0.110, 0.076, 0.101, 0.094])
# ))
# fig.update_layout(
#     barmode='group',
#     xaxis=dict(
#         title=dict(
#             text='Survival model'
#         ),
#         title_font=dict(size=20),
#         tickangle=0,
#         tickfont=dict(
#         size=20
#         ),
#     ),
#     yaxis=dict(
#         title=dict(
#             text='C-AUC'
#         ),
#         title_font=dict(size=20)
#     ),
#     margin=dict(t=10, l=50, r=30),
#     legend=dict(
#         x=0.01, y=0.01,
#         xanchor='left', yanchor='bottom',
#         font=dict(size=20)
#     )
# )
# fig.write_image(f"a_07explainable_AI/survival_metric_cauc.png", width=1000, height=500, scale=2)

# # plot IBS
# fig = go.Figure()
# fig.add_trace(go.Bar(
#     name='Pyradiomics',
#     x=['RSF', 'CoxPH', 'Coxnet', 'Avg'], y=[0.209, 0.239, 0.269, 0.239],
#     error_y=dict(type='data', array=[0.026, 0.044, 0.046, 0.039])
# ))
# fig.add_trace(go.Bar(
#     name='M3D-CLIP',
#     x=['RSF', 'CoxPH', 'Coxnet', 'Avg'], y=[0.193, 0.193, 0.201, 0.196],
#     error_y=dict(type='data', array=[0.026, 0.026, 0.028, 0.027])
# ))
# fig.add_trace(go.Bar(
#     name='SegVol ViT + Mean',
#     x=['RSF', 'CoxPH', 'Coxnet', 'Avg'], y=[0.199, 0.195, 0.187, 0.194],
#     error_y=dict(type='data', array=[0.033, 0.025, 0.025, 0.028])
# ))
# fig.add_trace(go.Bar(
#     name='SegVol ViT + ABMIL',
#     x=['RSF', 'CoxPH', 'Coxnet', 'Avg'], y=[0.190, 0.206, 0.191, 0.196],
#     error_y=dict(type='data', array=[0.028, 0.030, 0.026, 0.028])
# ))
# fig.add_trace(go.Bar(
#     name='SegVol ViT + SPARRA',
#     x=['RSF', 'CoxPH', 'Coxnet', 'Avg'], y=[0.189, 0.227, 0.190, 0.202],
#     error_y=dict(type='data', array=[0.023, 0.078, 0.022, 0.041])
# ))
# fig.update_layout(
#     barmode='group',
#     xaxis=dict(
#         title=dict(
#             text='Survival model'
#         ),
#         title_font=dict(size=20),
#         tickangle=0,
#         tickfont=dict(
#         size=20
#         ),
#     ),
#     yaxis=dict(
#         title=dict(
#             text='IBS'
#         ),
#         title_font=dict(size=20)
#     ),
#     margin=dict(t=10, l=50, r=30),
#     legend=dict(
#         x=0.01, y=0.01,
#         xanchor='left', yanchor='bottom',
#         font=dict(size=20)
#     )
# )
# fig.write_image(f"a_07explainable_AI/survival_metric_ibs.png", width=1000, height=500, scale=2)

# # pathomics
# # plot c index
# fig = go.Figure()
# fig.add_trace(go.Bar(
#     name='HIPT + Mean',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.604, 0.578, 0.577, 0.566, 0.581],
#     error_y=dict(type='data', array=[0.061, 0.095, 0.072, 0.073, 0.075])
# ))
# fig.add_trace(go.Bar(
#     name='UNI + Mean',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.616, 0.615, 0.626, 0.622, 0.620],
#     error_y=dict(type='data', array=[0.058, 0.064, 0.117, 0.073, 0.078])
# ))
# fig.add_trace(go.Bar(
#     name='CONCH + Mean',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.609, 0.614, 0.637, 0.600, 0.615],
#     error_y=dict(type='data', array=[0.083, 0.097, 0.091, 0.082, 0.088])
# ))
# fig.add_trace(go.Bar(
#     name='CHIEF + Mean',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.594, 0.577, 0.582, 0.544, 0.574],
#     error_y=dict(type='data', array=[0.103, 0.059, 0.049, 0.045, 0.064])
# ))
# fig.add_trace(go.Bar(
#     name='UNI + ABMIL',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.593, 0.612, 0.616, 0.609, 0.608],
#     error_y=dict(type='data', array=[0.105, 0.109, 0.071, 0.108, 0.098])
# ))
# fig.add_trace(go.Bar(
#     name='UNI + SPARRA',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.702, 0.690, 0.695, 0.717, 0.701],
#     error_y=dict(type='data', array=[0.077, 0.045, 0.064, 0.047, 0.058])
# ))
# fig.update_layout(
#     barmode='group',
#     xaxis=dict(
#         title=dict(
#             text='Survival model'
#         ),
#         title_font=dict(size=20),
#         tickangle=0,
#         tickfont=dict(
#         size=20
#         ),
#     ),
#     yaxis=dict(
#         title=dict(
#             text='C-Index'
#         ),
#         title_font=dict(size=20)
#     ),
#     margin=dict(t=10, l=50, r=30),
#     legend=dict(
#         x=0.01, y=0.01,
#         xanchor='left', yanchor='bottom',
#         font=dict(size=20)
#     )
# )
# fig.write_image(f"a_07explainable_AI/survival_metric_cindex.png", width=1000, height=500, scale=2)

# # plot c ipcw
# fig = go.Figure()
# fig.add_trace(go.Bar(
#     name='HIPT + Mean',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.575, 0.631, 0.643, 0.720, 0.642],
#     error_y=dict(type='data', array=[0.173, 0.157, 0.159, 0.030, 0.130])
# ))
# fig.add_trace(go.Bar(
#     name='UNI + Mean',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.635, 0.728, 0.658, 0.747, 0.692],
#     error_y=dict(type='data', array=[0.174, 0.078, 0.202, 0.058, 0.128])
# ))
# fig.add_trace(go.Bar(
#     name='CONCH + Mean',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.590, 0.640, 0.659, 0.635, 0.631],
#     error_y=dict(type='data', array=[0.154, 0.163, 0.172, 0.183, 0.168])
# ))
# fig.add_trace(go.Bar(
#     name='CHIEF + Mean',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.613, 0.632, 0.583, 0.610, 0.610],
#     error_y=dict(type='data', array=[0.198, 0.183, 0.172, 0.182, 0.184])
# ))
# fig.add_trace(go.Bar(
#     name='UNI + ABMIL',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.621, 0.645, 0.735, 0.745, 0.687],
#     error_y=dict(type='data', array=[0.187, 0.203, 0.079, 0.052, 0.130])
# ))
# fig.add_trace(go.Bar(
#     name='UNI + SPARRA',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.625, 0.709, 0.719, 0.740, 0.698],
#     error_y=dict(type='data', array=[0.196, 0.119, 0.100, 0.112, 0.132])
# ))
# fig.update_layout(
#     barmode='group',
#     xaxis=dict(
#         title=dict(
#             text='Survival model'
#         ),
#         title_font=dict(size=20),
#         tickangle=0,
#         tickfont=dict(
#         size=20
#         ),
#     ),
#     yaxis=dict(
#         title=dict(
#             text='C-IPCW'
#         ),
#         title_font=dict(size=20)
#     ),
#     margin=dict(t=10, l=50, r=30),
#     legend=dict(
#         x=0.01, y=0.01,
#         xanchor='left', yanchor='bottom',
#         font=dict(size=20)
#     )
# )
# fig.write_image(f"a_07explainable_AI/survival_metric_cipcw.png", width=1000, height=500, scale=2)

# # plot c auc
# fig = go.Figure()
# fig.add_trace(go.Bar(
#     name='HIPT + Mean',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.608, 0.595, 0.588, 0.574, 0.591],
#     error_y=dict(type='data', array=[0.031, 0.094, 0.062, 0.069, 0.064])
# ))
# fig.add_trace(go.Bar(
#     name='UNI + Mean',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.618, 0.610, 0.631, 0.620, 0.620],
#     error_y=dict(type='data', array=[0.070, 0.100, 0.142, 0.104, 0.104])
# ))
# fig.add_trace(go.Bar(
#     name='CONCH + Mean',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.606, 0.598, 0.613, 0.583, 0.600],
#     error_y=dict(type='data', array=[0.097, 0.127, 0.130, 0.124, 0.120])
# ))
# fig.add_trace(go.Bar(
#     name='CHIEF + Mean',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.586, 0.575, 0.577, 0.533, 0.568],
#     error_y=dict(type='data', array=[0.141, 0.073, 0.067, 0.068, 0.087])
# ))
# fig.add_trace(go.Bar(
#     name='UNI + ABMIL',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.574, 0.603, 0.608, 0.606, 0.598],
#     error_y=dict(type='data', array=[0.127, 0.144, 0.098, 0.142, 0.128])
# ))
# fig.add_trace(go.Bar(
#     name='UNI + SPARRA',
#     x=['RSF', 'CoxPH', 'Coxnet', 'FSVM', 'Avg'], y=[0.690, 0.680, 0.668, 0.705, 0.686],
#     error_y=dict(type='data', array=[0.078, 0.065, 0.098, 0.056, 0.074])
# ))
# fig.update_layout(
#     barmode='group',
#     xaxis=dict(
#         title=dict(
#             text='Survival model'
#         ),
#         title_font=dict(size=20),
#         tickangle=0,
#         tickfont=dict(
#         size=20
#         ),
#     ),
#     yaxis=dict(
#         title=dict(
#             text='C-AUC'
#         ),
#         title_font=dict(size=20)
#     ),
#     margin=dict(t=10, l=50, r=30),
#     legend=dict(
#         x=0.01, y=0.01,
#         xanchor='left', yanchor='bottom',
#         font=dict(size=20)
#     )
# )
# fig.write_image(f"a_07explainable_AI/survival_metric_cauc.png", width=1000, height=500, scale=2)

# # plot IBS
# fig = go.Figure()
# fig.add_trace(go.Bar(
#     name='HIPT + Mean',
#     x=['RSF', 'CoxPH', 'Coxnet', 'Avg'], y=[0.195, 0.206, 0.206, 0.202],
#     error_y=dict(type='data', array=[0.026, 0.022, 0.019, 0.022])
# ))
# fig.add_trace(go.Bar(
#     name='UNI + Mean',
#     x=['RSF', 'CoxPH', 'Coxnet', 'Avg'], y=[0.191, 0.252, 0.280, 0.241],
#     error_y=dict(type='data', array=[0.017, 0.093, 0.060, 0.057])
# ))
# fig.add_trace(go.Bar(
#     name='CONCH + Mean',
#     x=['RSF', 'CoxPH', 'Coxnet', 'Avg'], y=[0.198, 0.198, 0.204, 0.200],
#     error_y=dict(type='data', array=[0.019, 0.021, 0.069, 0.036])
# ))
# fig.add_trace(go.Bar(
#     name='CHIEF + Mean',
#     x=['RSF', 'CoxPH', 'Coxnet', 'Avg'], y=[0.189, 0.215, 0.249, 0.218],
#     error_y=dict(type='data', array=[0.021, 0.039, 0.032, 0.031])
# ))
# fig.add_trace(go.Bar(
#     name='UNI + ABMIL',
#     x=['RSF', 'CoxPH', 'Coxnet', 'Avg'], y=[0.219, 0.235, 0.272, 0.242],
#     error_y=dict(type='data', array=[0.025, 0.031, 0.048, 0.035])
# ))
# fig.add_trace(go.Bar(
#     name='UNI + SPARRA',
#     x=['RSF', 'CoxPH', 'Coxnet', 'Avg'], y=[0.198, 0.243, 0.203, 0.215],
#     error_y=dict(type='data', array=[0.051, 0.095, 0.071, 0.072])
# ))
# fig.update_layout(
#     barmode='group',
#     xaxis=dict(
#         title=dict(
#             text='Survival model'
#         ),
#         title_font=dict(size=20),
#         tickangle=0,
#         tickfont=dict(
#         size=20
#         ),
#     ),
#     yaxis=dict(
#         title=dict(
#             text='IBS'
#         ),
#         title_font=dict(size=20)
#     ),
#     margin=dict(t=10, l=50, r=30),
#     legend=dict(
#         x=0.01, y=0.01,
#         xanchor='left', yanchor='bottom',
#         font=dict(size=20)
#     )
# )
# fig.write_image(f"a_07explainable_AI/survival_metric_ibs.png", width=1000, height=500, scale=2)

# radiopathomics
aggr = ['Mean', 'ABMIL', 'SPARRA']
surv = ['RSF', 'CoxPH', 'Coxnet', 'FSVM']
x = []
for a in aggr:
    for s in surv:
        x.append(f"{a}+{s}")
# plot c index
fig = go.Figure()
y_mean = [0.575, 0.593, 0.639, 0.591, 0.623, 0.610, 0.624, 0.614, 0.639, 0.644, 0.647, 0.643]
y_std = [0.053, 0.061, 0.077, 0.076, 0.036, 0.063, 0.034, 0.078, 0.084, 0.088, 0.071, 0.089]
y_upper = (np.array(y_mean) + np.array(y_std)).tolist()
y_lower = (np.array(y_mean) - np.array(y_std)).tolist()
fig.add_trace(go.Scatter(
    name='Radiomics',
    x=x, y=y_mean,
    mode='lines+markers',
    line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    name='Upper Bound',
    x=x, y=y_upper,
    mode='lines',
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    name='Lower Bound',
    x=x, y=y_lower,
    line=dict(width=0),
    mode='lines',
    fillcolor='rgba(0,0,255,0.2)',
    fill='tonexty',
    showlegend=False
))
y_mean = [0.616, 0.615, 0.626, 0.622, 0.593, 0.612, 0.616, 0.609, 0.702, 0.690, 0.695, 0.717]
y_std = [0.058, 0.064, 0.117, 0.073, 0.105, 0.109, 0.071, 0.108, 0.077, 0.045, 0.064, 0.047]
y_upper = (np.array(y_mean) + np.array(y_std)).tolist()
y_lower = (np.array(y_mean) - np.array(y_std)).tolist()
fig.add_trace(go.Scatter(
    name='Pathomics',
    x=x, y=y_mean,
    mode='lines+markers',
    line=dict(color='green')
))
fig.add_trace(go.Scatter(
    name='Upper Bound',
    x=x, y=y_upper,
    mode='lines',
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    name='Lower Bound',
    x=x, y=y_lower,
    line=dict(width=0),
    mode='lines',
    fillcolor='rgba(0,255,0,0.2)',
    fill='tonexty',
    showlegend=False
))
y_mean = [0.614, 0.635, 0.614, 0.630, 0.597, 0.624, 0.614, 0.623, 0.678, 0.720, 0.693, 0.710]
y_std = [0.085, 0.077, 0.122, 0.086, 0.091, 0.093, 0.083, 0.091, 0.085, 0.068, 0.057, 0.071]
y_upper = (np.array(y_mean) + np.array(y_std)).tolist()
y_lower = (np.array(y_mean) - np.array(y_std)).tolist()
fig.add_trace(go.Scatter(
    name='Radiopathomics',
    x=x, y=y_mean,
    mode='lines+markers',
    line=dict(color='red')
))
fig.add_trace(go.Scatter(
    name='Upper Bound',
    x=x, y=y_upper,
    mode='lines',
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    name='Lower Bound',
    x=x, y=y_lower,
    line=dict(width=0),
    mode='lines',
    fillcolor='rgba(255,0,0,0.2)',
    fill='tonexty',
    showlegend=False
))
fig.update_layout(
    xaxis=dict(
        title=dict(
            text='Aggregation method + Survival model'
        ),
        title_font=dict(size=20),
        tickangle=30,
        tickfont=dict(
        size=20
        ),
    ),
    yaxis=dict(
        title=dict(
            text='C-Index'
        ),
        title_font=dict(size=20)
    ),
    margin=dict(t=10, l=50, r=30),
    legend=dict(
        x=0.01, y=0.99,
        xanchor='left', yanchor='top',
        font=dict(size=20)
    )
)
fig.write_image(f"a_07explainable_AI/survival_metric_cindex.png", width=1000, height=500, scale=2)

# plot c ipcw
fig = go.Figure()
y_mean = [0.575, 0.697, 0.749, 0.705, 0.679, 0.736, 0.754, 0.750, 0.574, 0.723, 0.755, 0.735]
y_std = [0.167, 0.100, 0.072, 0.097, 0.159, 0.062, 0.066, 0.032, 0.148, 0.097, 0.064, 0.088]
y_upper = (np.array(y_mean) + np.array(y_std)).tolist()
y_lower = (np.array(y_mean) - np.array(y_std)).tolist()
fig.add_trace(go.Scatter(
    name='Radiomics',
    x=x, y=y_mean,
    mode='lines+markers',
    line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    name='Upper Bound',
    x=x, y=y_upper,
    mode='lines',
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    name='Lower Bound',
    x=x, y=y_lower,
    line=dict(width=0),
    mode='lines',
    fillcolor='rgba(0,0,255,0.2)',
    fill='tonexty',
    showlegend=False
))
y_mean = [0.635, 0.728, 0.658, 0.747, 0.621, 0.645, 0.735, 0.745, 0.625, 0.709, 0.719, 0.740]
y_std = [0.174, 0.078, 0.202, 0.058, 0.187, 0.203, 0.079, 0.052, 0.196, 0.119, 0.100, 0.112]
y_upper = (np.array(y_mean) + np.array(y_std)).tolist()
y_lower = (np.array(y_mean) - np.array(y_std)).tolist()
fig.add_trace(go.Scatter(
    name='Pathomics',
    x=x, y=y_mean,
    mode='lines+markers',
    line=dict(color='green')
))
fig.add_trace(go.Scatter(
    name='Upper Bound',
    x=x, y=y_upper,
    mode='lines',
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    name='Lower Bound',
    x=x, y=y_lower,
    line=dict(width=0),
    mode='lines',
    fillcolor='rgba(0,255,0,0.2)',
    fill='tonexty',
    showlegend=False
))
y_mean = [0.640, 0.746, 0.759, 0.746, 0.612, 0.764, 0.640, 0.750, 0.625, 0.753, 0.727, 0.754]
y_std = [0.178, 0.065, 0.037, 0.067, 0.188, 0.052, 0.201, 0.055, 0.173, 0.085, 0.101, 0.086]
y_upper = (np.array(y_mean) + np.array(y_std)).tolist()
y_lower = (np.array(y_mean) - np.array(y_std)).tolist()
fig.add_trace(go.Scatter(
    name='Radiopathomics',
    x=x, y=y_mean,
    mode='lines+markers',
    line=dict(color='red')
))
fig.add_trace(go.Scatter(
    name='Upper Bound',
    x=x, y=y_upper,
    mode='lines',
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    name='Lower Bound',
    x=x, y=y_lower,
    line=dict(width=0),
    mode='lines',
    fillcolor='rgba(255,0,0,0.2)',
    fill='tonexty',
    showlegend=False
))
fig.update_layout(
    xaxis=dict(
        title=dict(
            text='Aggregation method + Survival model'
        ),
        title_font=dict(size=20),
        tickangle=30,
        tickfont=dict(
        size=20
        ),
    ),
    yaxis=dict(
        title=dict(
            text='C-IPCW'
        ),
        title_font=dict(size=20)
    ),
    margin=dict(t=10, l=50, r=30),
    legend=dict(
        x=0.99, y=0.01,
        xanchor='right', yanchor='bottom',
        font=dict(size=20)
    )
)
fig.write_image(f"a_07explainable_AI/survival_metric_cipcw.png", width=1000, height=500, scale=2)

# plot c auc
fig = go.Figure()
y_mean = [0.584, 0.594, 0.637, 0.595, 0.616, 0.612, 0.610, 0.613, 0.647, 0.632, 0.659, 0.645]
y_std = [0.057, 0.059, 0.078, 0.070, 0.057, 0.079, 0.025, 0.096, 0.090, 0.110, 0.076, 0.101]
y_upper = (np.array(y_mean) + np.array(y_std)).tolist()
y_lower = (np.array(y_mean) - np.array(y_std)).tolist()
fig.add_trace(go.Scatter(
    name='Radiomics',
    x=x, y=y_mean,
    mode='lines+markers',
    line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    name='Upper Bound',
    x=x, y=y_upper,
    mode='lines',
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    name='Lower Bound',
    x=x, y=y_lower,
    line=dict(width=0),
    mode='lines',
    fillcolor='rgba(0,0,255,0.2)',
    fill='tonexty',
    showlegend=False
))
y_mean = [0.618, 0.610, 0.631, 0.620, 0.574, 0.603, 0.608, 0.606, 0.690, 0.680, 0.668, 0.705]
y_std = [0.070, 0.100, 0.142, 0.104, 0.127, 0.144, 0.098, 0.142, 0.078, 0.065, 0.098, 0.056]
y_upper = (np.array(y_mean) + np.array(y_std)).tolist()
y_lower = (np.array(y_mean) - np.array(y_std)).tolist()
fig.add_trace(go.Scatter(
    name='Pathomics',
    x=x, y=y_mean,
    mode='lines+markers',
    line=dict(color='green')
))
fig.add_trace(go.Scatter(
    name='Upper Bound',
    x=x, y=y_upper,
    mode='lines',
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    name='Lower Bound',
    x=x, y=y_lower,
    line=dict(width=0),
    mode='lines',
    fillcolor='rgba(0,255,0,0.2)',
    fill='tonexty',
    showlegend=False
))
y_mean = [0.619, 0.628, 0.621, 0.627, 0.567, 0.620, 0.604, 0.621, 0.673, 0.718, 0.676, 0.708]
y_std = [0.104, 0.113, 0.157, 0.122, 0.132, 0.130, 0.106, 0.136, 0.084, 0.066, 0.084, 0.066]
y_upper = (np.array(y_mean) + np.array(y_std)).tolist()
y_lower = (np.array(y_mean) - np.array(y_std)).tolist()
fig.add_trace(go.Scatter(
    name='Radiopathomics',
    x=x, y=y_mean,
    mode='lines+markers',
    line=dict(color='red')
))
fig.add_trace(go.Scatter(
    name='Upper Bound',
    x=x, y=y_upper,
    mode='lines',
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    name='Lower Bound',
    x=x, y=y_lower,
    line=dict(width=0),
    mode='lines',
    fillcolor='rgba(255,0,0,0.2)',
    fill='tonexty',
    showlegend=False
))
fig.update_layout(
    xaxis=dict(
        title=dict(
            text='Aggregation method + Survival model'
        ),
        title_font=dict(size=20),
        tickangle=30,
        tickfont=dict(
        size=20
        ),
    ),
    yaxis=dict(
        title=dict(
            text='C-AUC'
        ),
        title_font=dict(size=20)
    ),
    margin=dict(t=10, l=50, r=30),
    legend=dict(
        x=0.99, y=0.01,
        xanchor='right', yanchor='bottom',
        font=dict(size=20)
    )
)
fig.write_image(f"a_07explainable_AI/survival_metric_cauc.png", width=1000, height=500, scale=2)

# plot ibs
aggr = ['Mean', 'ABMIL', 'SPARRA']
surv = ['RSF', 'CoxPH', 'Coxnet']
x = []
for a in aggr:
    for s in surv:
        x.append(f"{a}+{s}")
fig = go.Figure()
y_mean = [0.199, 0.195, 0.187, 0.190, 0.206, 0.191, 0.189, 0.227, 0.190]
y_std = [0.033, 0.025, 0.025, 0.028, 0.030, 0.026, 0.023, 0.078, 0.022]
y_upper = (np.array(y_mean) + np.array(y_std)).tolist()
y_lower = (np.array(y_mean) - np.array(y_std)).tolist()
fig.add_trace(go.Scatter(
    name='Radiomics',
    x=x, y=y_mean,
    mode='lines+markers',
    line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    name='Upper Bound',
    x=x, y=y_upper,
    mode='lines',
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    name='Lower Bound',
    x=x, y=y_lower,
    line=dict(width=0),
    mode='lines',
    fillcolor='rgba(0,0,255,0.2)',
    fill='tonexty',
    showlegend=False
))
y_mean = [0.191, 0.252, 0.280, 0.219, 0.235, 0.272, 0.198, 0.243, 0.203]
y_std = [0.017, 0.093, 0.060, 0.025, 0.031, 0.048, 0.051, 0.095, 0.071]
y_upper = (np.array(y_mean) + np.array(y_std)).tolist()
y_lower = (np.array(y_mean) - np.array(y_std)).tolist()
fig.add_trace(go.Scatter(
    name='Pathomics',
    x=x, y=y_mean,
    mode='lines+markers',
    line=dict(color='green')
))
fig.add_trace(go.Scatter(
    name='Upper Bound',
    x=x, y=y_upper,
    mode='lines',
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    name='Lower Bound',
    x=x, y=y_lower,
    line=dict(width=0),
    mode='lines',
    fillcolor='rgba(0,255,0,0.2)',
    fill='tonexty',
    showlegend=False
))
y_mean = [0.191, 0.293, 0.283, 0.225, 0.247, 0.271, 0.194, 0.203, 0.207]
y_std = [0.019, 0.083, 0.061, 0.028, 0.032, 0.043, 0.039, 0.049, 0.060]
y_upper = (np.array(y_mean) + np.array(y_std)).tolist()
y_lower = (np.array(y_mean) - np.array(y_std)).tolist()
fig.add_trace(go.Scatter(
    name='Radiopathomics',
    x=x, y=y_mean,
    mode='lines+markers',
    line=dict(color='red')
))
fig.add_trace(go.Scatter(
    name='Upper Bound',
    x=x, y=y_upper,
    mode='lines',
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    name='Lower Bound',
    x=x, y=y_lower,
    line=dict(width=0),
    mode='lines',
    fillcolor='rgba(255,0,0,0.2)',
    fill='tonexty',
    showlegend=False
))
fig.update_layout(
    xaxis=dict(
        title=dict(
            text='Aggregation method + Survival model'
        ),
        title_font=dict(size=20),
        tickangle=30,
        tickfont=dict(
        size=20
        ),
    ),
    yaxis=dict(
        title=dict(
            text='IBS'
        ),
        title_font=dict(size=20)
    ),
    margin=dict(t=10, l=50, r=30),
    legend=dict(
        x=0.99, y=0.99,
        xanchor='right', yanchor='top',
        font=dict(size=20)
    )
)
fig.write_image(f"a_07explainable_AI/survival_metric_ibs.png", width=1000, height=500, scale=2)