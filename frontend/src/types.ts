export type Session = {
  id: string;
  title: string;
  date: number;
  record: boolean;
  time_limit: number;
  description: string;
  creation_time: number;
  end_time: number;
  start_time: number;
  notes: Note[];
  participants: Participant[];
  log: [];
};

export type Note = {
  time: number;
  speakers: string[];
  content: string;
};

export type Participant = {
  id: string;
  participant_name: string;
  banned: boolean;
  size: { width: number; height: number };
  muted_video: boolean;
  muted_audio: boolean;
  position: { x: number; y: number; z: number };
  chat: Chat[];
  audio_filters: Filter[];
  video_filters: Filter[];
};

export type Box = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type Shape = {
  x: number;
  y: number;
  fill: string;
  participant_name: string;
};

export type Group = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type Chat = {
  message: string;
  time: number;
  author: string;
  target: string;
};

export type Filter = {
  id: string;
  type: string;
  channel: string;
  groupFilter: boolean;
  config: object;
};

type FilterConfig = {
  [key: string]: FilterConfigArray | FilterConfigNumber;
};

type FilterConfigNumber = {
  min: number;
  max: number;
  step: number;
  value: number;
  defaultValue: number;
};

type FilterConfigArray = {
  value: string;
  defaultValue: string[];
};
