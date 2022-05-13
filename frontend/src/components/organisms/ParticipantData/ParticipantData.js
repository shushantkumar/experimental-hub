import Button from "../../atoms/Button/Button";
import InputTextField from "../../molecules/InputTextField/InputTextField";
import "./ParticipantData.css";
import { FaRegTrashAlt } from "react-icons/fa";
import Checkbox from "../../molecules/Checkbox/Checkbox";
import Heading from "../../atoms/Heading/Heading";
import Label from "../../atoms/Label/Label";

function ParticipantData({
  onDeleteParticipant,
  onChange,
  index,
  first_name,
  last_name,
  link,
  muted_audio,
  muted_video,
  parameters,
  showParticipantInput,
  setShowParticipantInput,
}) {
  const handleChange = (
    first_name,
    last_name,
    link,
    muted_audio,
    muted_video
  ) => {
    onChange(index, {
      first_name,
      last_name,
      link,
      muted_audio,
      muted_video,
    });
  };

  const onAddAdditionalInformation = () => {
    //TODO: send data to backend and wait for link...

    setShowParticipantInput(!showParticipantInput);
  };

  return (
    <div className="participantDataContainer">
      <InputTextField
        title="Participant Name"
        placeholder={"Enter the information"}
        value={[first_name, last_name]
          .filter((str) => str.length > 0)
          .join(" ")}
        readonly={true}
      />
      <div className="participantButtons">
        <Button
          name="Enter participant information"
          design={"secondary"}
          onClick={() => onAddAdditionalInformation()}
        />

        <Button
          name={""}
          design={"negative"}
          onClick={() => onDeleteParticipant()}
          icon={<FaRegTrashAlt />}
        />
      </div>

      {showParticipantInput && (
        <div className="additionalParticipantInfoContainer">
          <div className="additionalParticipantInfo">
            <div className="additionalParticipantInfoCard">
              <Heading heading={"General information:"} />

              <InputTextField
                title="First Name"
                value={first_name}
                placeholder={"Name of participant"}
                onChange={(newFirstName) =>
                  handleChange(
                    newFirstName,
                    last_name,
                    link,
                    muted_audio,
                    muted_video
                  )
                }
              />
              <InputTextField
                title="Last Name"
                value={last_name}
                placeholder={"Name of participant"}
                onChange={(newLastName) =>
                  handleChange(
                    first_name,
                    newLastName,
                    link,
                    muted_audio,
                    muted_video
                  )
                }
              />
              <InputTextField
                title="Link"
                value={link}
                placeholder={"Save session to generate link."}
                readonly={true}
                onChange={(newLink) =>
                  handleChange(
                    first_name,
                    last_name,
                    newLink,
                    muted_audio,
                    muted_video
                  )
                }
              />
              <div className="participantMuteCheckbox">
                <Checkbox
                  title="Mute Audio"
                  value={muted_audio}
                  checked={muted_audio}
                  onChange={() =>
                    handleChange(
                      first_name,
                      last_name,
                      link,
                      !muted_audio,
                      muted_video
                    )
                  }
                />
                <Checkbox
                  title="Mute Video"
                  value={muted_video}
                  checked={muted_video}
                  onChange={() =>
                    handleChange(
                      first_name,
                      last_name,
                      link,
                      muted_audio,
                      !muted_video
                    )
                  }
                />
              </div>
              <Heading heading={"Current video position and size:"} />
              <div className="participantVideoSize">
                <div className="participantPosition">
                  <Label title={"x: "} /> {parameters.x}
                </div>
                <div className="participantPosition">
                  <Label title={"y: "} /> {parameters.y}
                </div>
                <div className="participantPosition">
                  <Label title={"Width: "} /> {parameters.width}
                </div>
                <div className="participantPosition">
                  <Label title={"Height: "} /> {parameters.height}
                </div>
              </div>
              <Button
                name="Save"
                design={"secondary"}
                onClick={() => onAddAdditionalInformation()}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ParticipantData;
