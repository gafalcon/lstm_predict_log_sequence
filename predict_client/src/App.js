import './App.css';
import React, { Component } from 'react';
import io from 'socket.io-client';
import { Bar } from 'react-chartjs-2';
import axios from 'axios';

const url = "http://localhost:5000"
class App extends Component {
    socket = null;

    data = {
        labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple'],
        datasets: [
            {
                label: 'Predictions',
                data: [12, 19, 3, 5, 2],
                backgroundColor: [
                    'rgba(255, 99, 132, 0.2)',
                    'rgba(54, 162, 235, 0.2)',
                    'rgba(255, 206, 86, 0.2)',
                    'rgba(75, 192, 192, 0.2)',
                    'rgba(153, 102, 255, 0.2)',
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(75, 192, 192, 1)',
                    'rgba(153, 102, 255, 1)',
                ],
                borderWidth: 1,
            },
        ],
    };

    options = {
        plugins: {
            title: {
                display: true,
                text: 'Probabilities of next action'
            },
        },
        scales: {
            yAxes: [
                {
                    ticks: {
                        beginAtZero: true,
                    },
                },
            ],
        },
    };

    constructor(props){
        super(props);
        this.state = {
            actions: [],
            predictions: [],
            probs: [],
            currentIdx: -1,
            logResults: {}
        }
        this.fileInput = React.createRef();
        this.openSocketConnection = this.openSocketConnection.bind(this);
        this.subscribeToNewActions = this.subscribeToNewActions.bind(this);
        this.setNewAction = this.setNewAction.bind(this)
        this.setIdx = this.setIdx.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
    }

    setNewAction(newAction){
        this.setState((prevState)=>{
            return {
                actions: [...prevState.actions, newAction.action],
                predictions: [...prevState.predictions, newAction.pred],
                probs: [...prevState.probs, newAction.p],
                currentIdx: prevState.actions.length
            }
        })
    }

    openSocketConnection(){
        if (!this.socket){
            this.socket = io.connect(url);
            this.socket.on('connect', ()=> {
                console.log("socket connected")
                this.subscribeToNewActions()
            })
            this.socket.on("disconnect", ()=>{
                console.error("Socket disconnected")
                this.socket = null;
            })
            this.socket.on("error", (data)=>{
                this.socket = null;
                console.error(data || "socket error");
            })
        }
    }

    subscribeToNewActions(){
        if (this.socket){
            this.socket.on('new_action', (action)=> {
                this.setNewAction(action)
            })
            this.socket.on('first_action', (action)=>{
                this.setState({
                    actions: [action.action],
                    predictions: [action.pred],
                    probs: [action.p],
                    currentIdx: 0
                })
            })
        }
    }

    unsubscribeToNewActions(){
        if (this.socket){
            this.socket.removeAllListeners('new_action')
        }
    }

    componentDidMount(){
        this.openSocketConnection();
    }

    componentWillUnmount(){
        if (this.socket){
            this.socket.disconnect();
        }
    }

    setIdx(idx){
        if (idx >= 0 && idx < this.state.actions.length){
            this.setState({currentIdx: idx});
        }
    }

    handleSubmit(event){
        event.preventDefault();
        console.log(this.fileInput.current.files)
        let formData = new FormData();
        formData.append("logfile", this.fileInput.current.files[0]);
        axios.post("http://localhost:5000/predictSequence", formData, {
            headers: {
                "Content-Type": "multipart/form-data",
            },
        }).then((response) => {
            console.log(response.data)
            this.setState({logResults: {
                urls: response.data.urls,
                req_types: response.data.req_types,
                predictions: response.data.predictions,
                probs: response.data.probs
            }})
        }).catch((error) => {
            console.error(error)
        });
    }

    render(){
        let barData;
        if (this.state.currentIdx >= 0) {
            barData = {...this.data};
            barData['labels'] = this.state.predictions[this.state.currentIdx]
            barData['datasets'][0]['data'] = this.state.probs[this.state.currentIdx]
        }
        return (
            <div className="container">
                <h1>Predict KF Actions</h1>
                <div className="row">
                    <form onSubmit={this.handleSubmit}>
                        <div className="mb-3">
                            <label htmlFor="formFile" className="form-label">Submit log file to predict actions</label>
                            <input className="form-control" type="file" id="formFile" ref={this.fileInput}/>
                            <button className="btn btn-primary mt-1" type="submit">Submit</button>
                        </div>
                    </form>
                </div>
                { this.state.logResults.urls &&
                <div className="row">
                    <div className="col-4">
                        <h4>Logs</h4>
                        <ol>
                            {this.state.logResults.urls.map(
                                (url, i) => <li key={i}>{this.state.logResults.req_types[i]} {url}</li>)}
                        </ol>
                    </div>
                    <div className="col-8">
                        <h4>Predicted Actions</h4>
                        <table className="table">
                            <thead>
                                <tr>
                                    <th scope="col">#</th>
                                    <th scope="col">Action 1</th>
                                    <th scope="col">Action 2</th>
                                    <th scope="col">Action 3</th>
                                </tr>
                            </thead>
                            <tbody>

                                {this.state.logResults.predictions.map(
                                    (pred_array, i) => (
                                        <tr>
                                            <th scope="row">{i + 1}</th>
                                            {pred_array.map(
                                                (action, j) => <td>
                                                    {action}
                                                    ({Number(this.state.logResults.probs[i][j]).toFixed(3)})
                                                </td>
                                            )}
                                        </tr>
                                    )
                                )}
                            </tbody>
                        </table>

                    </div>
                </div>
                }
            { this.state.currentIdx >= 0 ?
              <div>
                  <h2>Real-time action predictions</h2>
                <div className="row">
                    <div className="col-lg-8 col-md-12">
                        <div className="row">
                            <Bar data={barData} options={this.options} />
                        </div>
                        <div className="row">
                            <button type="button" className="btn btn-sm btn-primary col m-1" onClick={()=>this.setIdx(0)}>&lt;&lt;</button>
                            <button type="button" className="btn btn-sm btn-primary col m-1" onClick={()=>this.setIdx(this.state.currentIdx - 1)}>&lt;</button>
                            <button type="button" className="btn btn-sm btn-primary col m-1" onClick={()=>this.setIdx(this.state.currentIdx + 1)}> &#62; </button>
                            <button type="button" className="btn btn-sm btn-primary col m-1" onClick={()=>this.setIdx(this.state.actions.length - 1)}> &#62;&#62; </button>
                        </div>
                    </div>
                    <div className="col-lg-4 col-md-8">
                        <h3>Actions:</h3>
                        <ol>
                            {
                                this.state.actions.slice(0, this.state.currentIdx +1).map((action) =>
                                <li>{ action }</li>
                            )
                            }
                        </ol>
                    </div>

                </div>
              </div>
              : ''
            }
            </div>
        );
    }
}

export default App;
