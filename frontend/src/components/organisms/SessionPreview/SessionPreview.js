import { integerToDateTime, isFutureSession } from "../../../utils/utils";
import Button from "../../atoms/Button/Button";
import LinkButton from "../../atoms/LinkButton/LinkButton";
import "./SessionPreview.css";

import { useDispatch } from "react-redux";
import { copySession, initializeSession } from "../../../features/openSession";
import Heading from "../../atoms/Heading/Heading";

function SessionPreview({
  selectedSession,
  setSelectedSession,
  onDeleteSession,
  onJoinExperiment,
  onCreateExperiment,
}) {
  console.log("selected session sessionPreview", selectedSession);
  const dispatch = useDispatch();
  const deleteSession = () => {
    const sessionId = selectedSession.id;
    onDeleteSession(sessionId);
    setSelectedSession(null);
  };

  return (
    <div
      className={
        "sessionPreviewContainer" +
        (selectedSession.creation_time > 0 && selectedSession.end_time === 0
          ? " ongoing"
          : "")
      }
    >
      <div className="sessionPreviewHeader">
        <div className="ongoingExperiment">
          {selectedSession.creation_time > 0 &&
            selectedSession.end_time === 0 && (
              <Heading heading={"Experiment ongoing."} />
            )}
        </div>
        <h3 className="sessionPreviewTitles">Title: {selectedSession.title}</h3>
        <h3 className="sessionPreviewTitles">
          Date: {integerToDateTime(selectedSession.date)}
        </h3>
        <h3 className="sessionPreviewTitles">
          Time Limit: {selectedSession.time_limit / 60000} minutes
        </h3>
      </div>
      <p className="sessionPreviewInformation">{selectedSession.description}</p>
      <>
        <div className="sessionPreviewButtons">
          {(selectedSession.creation_time === 0 ||
            selectedSession.end_time > 0) && (
            <Button
              name={"DELETE"}
              design={"negative"}
              onClick={() => deleteSession()}
            />
          )}
          <LinkButton
            name={"COPY"}
            to="/sessionForm"
            onClick={() => dispatch(copySession(selectedSession))}
          />
          {!selectedSession.creation_time > 0 &&
            selectedSession.end_time === 0 &&
            isFutureSession(selectedSession) && (
              <>
                <LinkButton
                  name={"EDIT"}
                  to="/sessionForm"
                  onClick={() => dispatch(initializeSession(selectedSession))}
                />
                <LinkButton
                  name={"START"}
                  to="/watchingRoom"
                  onClick={() => onCreateExperiment(selectedSession.id)}
                />
              </>
            )}
          {selectedSession.creation_time > 0 && selectedSession.end_time === 0 && (
            <>
              <LinkButton
                name={"JOIN EXPERIMENT"}
                to="/watchingRoom"
                onClick={() => onJoinExperiment(selectedSession.id)}
              />
            </>
          )}
        </div>
      </>
    </div>
  );
}

export default SessionPreview;
