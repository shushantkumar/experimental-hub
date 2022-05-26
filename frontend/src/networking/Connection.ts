import { BACKEND } from "../utils/constants";
import { EventHandler } from "./EventHandler";
import { isValidConnectionOffer, isValidMessage, Message } from "./MessageTypes";
import SubConnection from "./SubConnection";

export enum ConnectionState {
  NEW,
  CONNECTING,
  CONNECTED,
  CLOSED,
  FAILED
}

export default class Connection extends EventHandler<ConnectionState | MediaStream | MediaStream[]> {
  readonly api: EventHandler<any>;
  readonly sessionId?: string;
  readonly participantId?: string;
  readonly userType: "participant" | "experimenter";

  // Private variables
  private _state: ConnectionState;

  private localStream: MediaStream;
  private _remoteStream: MediaStream;
  private _peerStreams: Map<string, MediaStream>;

  private mainPc: RTCPeerConnection; // RTCPeerConnection | undefined
  private dc: RTCDataChannel;

  private subConnections: SubConnection[];


  constructor(userType: "participant" | "experimenter", sessionId?: string, participantId?: string) {
    super(true, "ConnectionEvents");
    if (userType === "participant" && (!participantId || !sessionId)) {
      throw new Error("[Connection] userType participant requires the participantId and sessionId to be defined.");
    }
    this.sessionId = sessionId;
    this.participantId = participantId;
    this.userType = userType;
    this.subConnections = [];
    this._state = ConnectionState.NEW;
    this._remoteStream = new MediaStream();
    this._peerStreams = new Map();

    this.api = new EventHandler();
    this.api.on("CONNECTION_OFFER", this.handleConnectionOffer.bind(this));

    this.initMainPeerConnection();
    this.initDataChannel();
  }

  public get remoteStream(): MediaStream {
    return this._remoteStream;
  }

  public get peerStreams(): MediaStream[] {
    return Array.from(this._peerStreams, ([_, value]) => value);
  }

  public get state(): ConnectionState {
    return this._state;
  }

  public async start(localStream?: MediaStream) {
    if (!localStream && this.userType === "participant") {
      throw new Error("Connection.start(): localStream is required for user type participant.");
    }
    if (this._state !== ConnectionState.NEW) {
      throw new Error(`Connection.start(): cannot start Connection, state is: ${ConnectionState[this._state]}`);
    }
    this.localStream = localStream;
    this.setState(ConnectionState.CONNECTING);

    // Add localStream to peer connection
    console.log("[Connection] Stating -- Adding localStream:", this.localStream);
    this.localStream?.getTracks().forEach((track) => {
      console.log("Adding track", track);
      this.mainPc.addTrack(track, this.localStream);
    });

    await this.negotiate();
  }

  public stop(closeSenders: boolean = true) {
    this.internalStop(undefined, closeSenders);
  }

  private internalStop(state?: ConnectionState, closeSenders: boolean = true) {
    if ([ConnectionState.CLOSED, ConnectionState.FAILED].includes(this._state)) {
      return;
    }
    this.setState(state ?? ConnectionState.CLOSED);

    console.log("[Connection] Stopping");

    this.subConnections.forEach(sc => sc.stop());

    this.dc.close();

    // close transceivers
    this.mainPc.getTransceivers().forEach(function (transceiver) {
      if (transceiver.currentDirection && transceiver.currentDirection !== "stopped") {
        transceiver.stop();
      }
    });

    // close local audio / video
    if (closeSenders) {
      this.mainPc.getSenders().forEach(function (sender) {
        if (sender && sender.track) {
          sender.track.stop();
        };
      });
    }

    // close peer connection
    this.mainPc.close();
  }

  public sendMessage(endpoint: string, data: any) {
    if (this._state !== ConnectionState.CONNECTED) {
      throw Error(`[Connection] Cannot send message if connection state is not Connected. State: ${ConnectionState[this._state]}`);
    }

    console.groupCollapsed(`[Connection] Sending ${endpoint} message`);
    console.log(data);
    console.groupEnd();

    const message: Message = {
      type: endpoint,
      data: data
    };
    const stringified = JSON.stringify(message);
    this.dc.send(stringified);
  }

  private setState(state: ConnectionState): void {
    this._state = state;
    this.trigger("connectionStateChange", state);
  }

  private initMainPeerConnection() {
    const config: any = {
      sdpSemantics: "unified-plan",
    };
    this.mainPc = new RTCPeerConnection(config);

    // register event listeners for pc
    this.mainPc.addEventListener(
      "icegatheringstatechange",
      () => console.log(`[Connection] icegatheringstatechange: ${this.mainPc.iceGatheringState}`),
      false
    );
    this.mainPc.addEventListener(
      "iceconnectionstatechange",
      this.handleIceConnectionStateChange.bind(this),
      false
    );
    this.mainPc.addEventListener(
      "signalingstatechange",
      this.handleSignalingStateChange.bind(this),
      false
    );

    // Receive audio / video
    this.mainPc.addEventListener("track", (e) => {
      console.groupCollapsed(`[Connection] Received ${e.track.kind} track from remote`);
      console.log(e);
      console.groupEnd();

      if (e.track.kind !== "video" && e.track.kind !== "audio") {
        console.error("[Connection] Received track with unknown kind:", e.track.kind);
        return;
      }

      this._remoteStream.addTrack(e.track);
      this.trigger("remoteStreamChange", this._remoteStream);
    });
  }

  private handleIceConnectionStateChange() {
    console.log(`[Connection] iceConnectionState: ${this.mainPc.iceConnectionState}`);
    if (["disconnected", "closed"].includes(this.mainPc.iceConnectionState)) {
      this.stop();
      return;
    }
    if (this.mainPc.iceConnectionState === "failed") {
      this.internalStop(ConnectionState.FAILED);
    }
  }

  private handleSignalingStateChange() {
    console.log(`[Connection] signalingState: ${this.mainPc.signalingState}`);
    if (this.mainPc.signalingState === "closed") {
      this.stop();
    }
  }

  private initDataChannel() {
    this.dc = this.mainPc.createDataChannel("API");
    this.dc.onclose = (_) => {
      console.log("[Connection] datachannel onclose");
      this.stop();
    };
    this.dc.onopen = (_) => {
      console.log("[Connection] datachannel onopen");
      this.setState(ConnectionState.CONNECTED);
    };
    this.dc.onmessage = this.handleDcMessage.bind(this);
  }

  private handleDcMessage(e: MessageEvent<any>) {
    let message;
    try {
      message = JSON.parse(e.data);
    } catch (error) {
      console.error("[Connection] Failed to parse datachannel message received from the server.");
      return;
    }
    if (!isValidMessage(message)) {
      console.error("[Connection] Received invalid message.", message);
      return;
    }
    console.log("[Connection] Received", message);
    this.api.trigger(message.type, message.data);
  }

  private async negotiate() {
    const offer = await this.mainPc.createOffer({
      offerToReceiveVideo: true,
      offerToReceiveAudio: true,
    });
    await this.mainPc.setLocalDescription(offer);

    // Wait for iceGatheringState to be "complete".
    await new Promise((resolve) => {
      if (this.mainPc?.iceGatheringState === "complete") {
        resolve(undefined);
      } else {
        const checkState = () => {
          if (this.mainPc?.iceGatheringState === "complete") {
            this.mainPc.removeEventListener(
              "icegatheringstatechange",
              checkState
            );
            resolve(undefined);
          }
        };
        this.mainPc?.addEventListener("icegatheringstatechange", checkState);
      }
    });

    const localDesc = this.mainPc.localDescription;
    let request;
    if (this.userType === "participant") {
      request = {
        sdp: localDesc.sdp,
        type: localDesc.type,
        user_type: "participant",
        session_id: this.sessionId,
        participant_id: this.participantId,
      };
    } else {
      request = {
        sdp: localDesc.sdp,
        type: localDesc.type,
        user_type: "experimenter",
      };
    }
    console.log("[Connection] Sending initial offer");

    let response;
    try {
      response = await fetch(BACKEND + "/offer", {
        body: JSON.stringify({ request }),
        headers: {
          "Content-Type": "application/json",
        },
        method: "POST",
        mode: "cors", // TODO for dev only
      });
    } catch (error) {
      console.error("[Connection] Failed to connect to backend.", error.message);
      this.setState(ConnectionState.FAILED);
      return;
    }

    if (!response.ok) {
      console.error("[Connection] Failed to connect to backend. Response not ok");
      this.setState(ConnectionState.FAILED);
      return;
    }


    const answer = await response.json();
    if (answer.type !== "SESSION_DESCRIPTION") {
      console.log(
        "[Connection] Received unexpected answer from backend. type:",
        answer.type
      );
      return;
    }

    console.log("[Connection] Received answer:", answer);

    const remoteDescription = answer.data;
    await this.mainPc.setRemoteDescription(remoteDescription);
  }

  private async handleConnectionOffer(data: any): Promise<void> {
    if (!isValidConnectionOffer(data)) {
      console.error("[Connection] received invalid CONNECTION_OFFER.");
      return;
    }
    const subConnection = new SubConnection(data, this);
    this._peerStreams.set(data.id, subConnection.remoteStream);
    subConnection.on("remoteStreamChange", async (_) => {
      this.trigger("remotePeerStreamsChange", this.peerStreams);
    });
    subConnection.on("connectionClosed", async (id) => {
      console.log("[Connection] Subconnection connectionClosed event triggered. Removing subConnection:", id);
      this._peerStreams.delete(id as string);
      this.trigger("remotePeerStreamsChange", this.peerStreams);
    });
    await subConnection.start();
    this.subConnections.push(subConnection);
  }
}
