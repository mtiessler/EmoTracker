import React, { JSX } from 'react';
import Plot2D from './Plot2D';
import Plot3D4D from './Plot3D4D';
import Plot2DVAD from './Plot2DVAD';
import Alert from 'react-bootstrap/Alert';
import Card from 'react-bootstrap/Card';
import Icon from '@mdi/react';
import { mdiAlertCircleOutline, mdiInformationOutline, mdiChartBoxOutline } from '@mdi/js';
import { WordData, WordSenses, SenseInfo, VizType, LoadedData } from '../types';
import { VADDimension } from '../vadUtils';

import '../styles/PlotContainer.scss';

interface PlotContainerProps {
    vizType: VizType;
    jsonData: LoadedData | null;
    selectedWords: string[];
    senseData: WordSenses | null;
    selectedSenseId: string;
}

const PlotContainer: React.FC<PlotContainerProps> = ({ vizType, jsonData, selectedWords, senseData, selectedSenseId }) => {

    const isMultiWord = selectedWords.length > 1;
    const firstSelectedWord = selectedWords[0];
    const firstWordData = jsonData && firstSelectedWord ? jsonData[firstSelectedWord] : null;

    if (selectedWords.length === 0) {
        return (
            <Card className="plot-card h-100">
                <Card.Header>Visualization</Card.Header>
                <Card.Body className="plot-container-body">
                    <div className="placeholder-content info">
                        <Icon path={mdiChartBoxOutline} size={1.6} className="placeholder-icon text-muted" />
                        <h4>Select Word(s)</h4>
                        <p>Select word(s) from the controls to display visualization.</p>
                    </div>
                </Card.Body>
                <Card.Footer className="plot-card-footer">
                    <div className="sense-info-placeholder">&nbsp;</div>
                </Card.Footer>
            </Card>
        );
    }

    const hasAnyValidData = selectedWords.some(word => {
        const wd = jsonData?.[word];
        return wd?.temporal_vad?.x?.length && wd?.temporal_vad?.v?.length && wd?.temporal_vad?.a?.length && wd?.temporal_vad?.d?.length;
    });

    if (!hasAnyValidData) {
        return (
            <Card className="plot-card h-100">
                <Card.Header className="error-header">Data Error</Card.Header>
                <Card.Body className="plot-container-body">
                    <div className="placeholder-content error-content">
                        <Icon path={mdiAlertCircleOutline} size={1.6} className="placeholder-icon" />
                        <h4>Data Incomplete</h4>
                        <p>Insufficient VAD data for selected word(s) to render plot.</p>
                    </div>
                </Card.Body>
                <Card.Footer className="plot-card-footer">
                    <div className="sense-info-placeholder">&nbsp;</div>
                </Card.Footer>
            </Card>
        );
    }

    const selectedSenseInfo: SenseInfo | undefined | null = !isMultiWord && senseData ? senseData[selectedSenseId] : null;

    let plotElement: JSX.Element | null = null;
    let plotTitle = selectedWords.join(', ');
    let noticeMessage: { type: 'info' | 'warning' | 'danger', text: string } | null = null;
    let placeholderContent: JSX.Element | null = null;

    const renderPlaceholder = (icon: string, title: string, text: string, type: 'info' | 'warning' | 'error' = 'info') => (
        <div className={`placeholder-content ${type}`}>
            <Icon path={icon} size={1.6} className="placeholder-icon" />
            <h4>{title}</h4>
            <p>{text}</p>
        </div>
    );

    try {
        switch (vizType) {
            case '2D-V':
            case '2D-A':
            case '2D-D':
                const yLabel: VADDimension = vizType === '2D-V' ? 'Valence' : vizType === '2D-A' ? 'Arousal' : 'Dominance';
                plotTitle = `${selectedWords.join(', ')}: ${yLabel} / Time`;
                plotElement = <Plot2D
                    selectedWords={selectedWords}
                    allWordsData={jsonData}
                    yLabel={yLabel}
                    selectedSenseData={!isMultiWord ? selectedSenseInfo : null}
                />;
                break;

            case '2D-VAD':
                plotTitle = `${selectedWords.join(', ')}: VAD / Time`;
                plotElement = <Plot2DVAD
                    selectedWords={selectedWords}
                    allWordsData={jsonData}
                    selectedSenseData={!isMultiWord ? selectedSenseInfo : null}
                />;
                break;

            case '3D':
                plotTitle = `${selectedWords.join(', ')}: VAD Trajectory (Time)`;
                plotElement = <Plot3D4D selectedWords={selectedWords} allWordsData={jsonData} is4D={false} />;
                break;

            case '4D':
                if (isMultiWord) {
                    plotTitle = `${selectedWords.join(', ')}: VAD Trajectory (Time)`;
                    plotElement = <Plot3D4D selectedWords={selectedWords} allWordsData={jsonData} is4D={false} />;
                    noticeMessage = { type: 'info', text: `4D Sense coloring disabled for multi-word view.` };
                } else if (selectedSenseInfo?.y_fitting && firstWordData?.temporal_vad?.x && selectedSenseInfo.y_fitting.length === firstWordData.temporal_vad.x.length) {
                    plotTitle = `${firstSelectedWord}: VAD Trajectory (Sense: ${selectedSenseId})`;
                    plotElement = <Plot3D4D selectedWords={selectedWords} allWordsData={jsonData} senseProportions={selectedSenseInfo.y_fitting} is4D={true} />;
                } else {
                    plotTitle = `${firstSelectedWord}: VAD Trajectory (Time)`;
                    plotElement = <Plot3D4D selectedWords={selectedWords} allWordsData={jsonData} is4D={false} />;
                    if (selectedSenseId && firstWordData) {
                        noticeMessage = { type: 'info', text: `Sense "${selectedSenseId}" lacks proportion data for 4D color.` };
                    } else if (!selectedSenseId && firstWordData) {
                        noticeMessage = { type: 'info', text: `Select a sense for 4D color visualization.` };
                    } else {
                        placeholderContent = renderPlaceholder(mdiAlertCircleOutline, 'Data Error', `Cannot load data for ${firstSelectedWord}`, 'error');
                    }
                }
                break;

            default:
                const exhaustiveCheck: never = vizType;
                placeholderContent = renderPlaceholder(mdiChartBoxOutline, 'Invalid Selection', 'Invalid visualization type selected.', 'error');
                noticeMessage = { type: 'danger', text: `Invalid visualization type: ${exhaustiveCheck}` };
        }
    } catch (error: any) {
        console.error(`Error rendering plot type ${vizType} for ${selectedWords.join(', ')}:`, error);
        placeholderContent = renderPlaceholder(mdiAlertCircleOutline, 'Rendering Error', error.message || 'Unknown error', 'error');
        noticeMessage = { type: 'danger', text: `Failed to render visualization: ${error.message || 'Unknown error'}` };
    }

    const shouldShowSenseInfo = !isMultiWord && selectedSenseInfo;

    const noticeIcon = noticeMessage?.type === 'danger' ? mdiAlertCircleOutline :
        noticeMessage?.type === 'warning' ? mdiAlertCircleOutline :
            mdiInformationOutline;

    return (
        <Card className="plot-card h-100">
            <Card.Header>{plotTitle}</Card.Header>
            <Card.Body className="plot-container-body">
                <div style={{ width: '100%', height: '100%' }}>
                    {plotElement ?? placeholderContent}
                </div>
                {noticeMessage && (
                    <Alert variant={noticeMessage.type} className="plot-notice d-flex align-items-center">
                        <Icon path={noticeIcon} size={0.9} className="me-2 flex-shrink-0" />
                        <span className="text-truncate">{noticeMessage.text}</span>
                    </Alert>
                )}
            </Card.Body>
            <Card.Footer className="plot-card-footer">
                {shouldShowSenseInfo ? (
                    <div className="sense-info-box">
                        <div><strong>Sense:</strong> <span>{selectedSenseId}</span></div>
                        <div><strong>Def:</strong> <span>{selectedSenseInfo.definition || '(No definition provided)'}</span></div>
                        <div><strong>Static VAD:</strong> <span>V={selectedSenseInfo.vad[0]?.toFixed(3) ?? 'N/A'}, A={selectedSenseInfo.vad[1]?.toFixed(3) ?? 'N/A'}, D={selectedSenseInfo.vad[2]?.toFixed(3) ?? 'N/A'}</span></div>
                    </div>
                ) : (
                    <div className="sense-info-placeholder">{isMultiWord ? 'Sense info disabled for multi-word view' : '\u00A0'}</div>
                )}
            </Card.Footer>
        </Card>
    );
};

export default PlotContainer;