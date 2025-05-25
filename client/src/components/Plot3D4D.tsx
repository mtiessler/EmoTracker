import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import type { Data, Layout, Font, ColorBar, Annotations } from 'plotly.js';
import Icon from '@mdi/react';
import { mdiAlertCircleOutline } from '@mdi/js';
import { LoadedData, PredictedVadPoint } from '../types';
import { getVADDescription } from '../vadUtils';

import '../styles/Plot3D4D.scss';

interface Plot3D4DProps {
    selectedWords: string[];
    allWordsData: LoadedData | null;
    senseProportions?: (number | null)[] | null;
    is4D?: boolean;
    predictedVadSeries?: PredictedVadPoint[] | null;
}

const createHoverTextWithDescPlain = (
    time: number[] | undefined,
    v: (number | null)[] | undefined,
    a: (number | null)[] | undefined,
    d: (number | null)[] | undefined,
    word: string,
    isForecast: boolean = false
): string[] => {
    if (!time || !v || !a || !d) return [];
    const forecastLabel = isForecast ? " (Forecast)" : "";
    return time.map((t, i) => {
        const vVal = v[i];
        const aVal = a[i];
        const dVal = d[i];
        const vDesc = getVADDescription('Valence', vVal);
        const aDesc = getVADDescription('Arousal', aVal);
        const dDesc = getVADDescription('Dominance', dVal);

        return `<b>Word: ${word}${forecastLabel} | Year: ${t}</b><br>` +
            `V: ${vVal?.toFixed(3) ?? 'N/A'} ${vDesc ? `(${vDesc})` : ''}<br>` +
            `A: ${aVal?.toFixed(3) ?? 'N/A'} ${aDesc ? `(${aDesc})` : ''}<br>` +
            `D: ${dVal?.toFixed(3) ?? 'N/A'} ${dDesc ? `(${dDesc})` : ''}`;
    });
};

const FONT_FAMILY = 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif';
const AXIS_TICK_COLOR = '#6c757d';
const AXIS_LABEL_COLOR = '#6c757d';
const AXIS_LINE_COLOR = '#dee2e6';
const AXIS_GRID_COLOR = '#f8f9fa';
const AXIS_BG_COLOR = '#e9ecef';
const PAPER_BG_COLOR = 'white';
const PLOT_BG_COLOR = 'white';
const TEXT_COLOR = '#212529';
const HOVER_BG_COLOR = 'white';
const HOVER_BORDER_COLOR = '#dee2e6';
const TIME_COLOR_SCALE = 'Blues';
const SENSE_COLOR_SCALE = 'Cividis';
const ANNOTATION_COLOR = '#495057';

const BASE_FONT: Partial<Font> = { family: FONT_FAMILY, size: 11, color: TEXT_COLOR };
const AXIS_TICK_FONT: Partial<Font> = { family: FONT_FAMILY, size: 9, color: AXIS_TICK_COLOR };
const AXIS_TITLE_FONT: Partial<Font> = { family: FONT_FAMILY, size: 10, color: AXIS_LABEL_COLOR };
const COLORBAR_TICK_FONT: Partial<Font> = { family: FONT_FAMILY, size: 9, color: AXIS_TICK_COLOR };
const HOVER_FONT: Partial<Font> = { family: FONT_FAMILY, size: 12.8, color: TEXT_COLOR };
const ANNOTATION_FONT: Partial<Font> = { family: FONT_FAMILY, size: 9, color: ANNOTATION_COLOR };

const traceColors = ['#0d6efd', '#dc3545', '#198754', '#ffc107', '#6f42c1', '#fd7e14', '#20c997', '#6610f2'];
const forecastLineStyle3D = { dash: 'dash' as const, width: 1.5 };


const Plot3D4D: React.FC<Plot3D4DProps> = ({ selectedWords, allWordsData, senseProportions = null, is4D = false, predictedVadSeries }) => {

    const plotMemoData = useMemo(() => {
        const traces: Data[] = [];
        let hasValidData = false;
        const annotations: Partial<Annotations>[] = [];
        const allV_plot: number[] = [];
        const allA_plot: number[] = [];
        const allD_plot: number[] = [];

        if (!allWordsData || selectedWords.length === 0) {
            return { traces, hasValidData, annotations, dataRanges: { x: [0,1] as [number,number], y: [0,1] as [number,number], z: [0,1] as [number,number] } };
        }

        const canShow4D = is4D && selectedWords.length === 1 && Array.isArray(senseProportions) && senseProportions.some(p => p !== null);

        selectedWords.forEach((word, index) => {
            const wordData = allWordsData[word];
            if (!wordData?.temporal_vad?.x || !wordData?.temporal_vad?.v || !wordData?.temporal_vad?.a || !wordData?.temporal_vad?.d) {
                console.warn(`Missing VAD data for word: ${word}`);
                return;
            }
            hasValidData = true;
            const { x: time, v, a, d } = wordData.temporal_vad;
            const traceColor = traceColors[index % traceColors.length];
            const dataLength = Math.min(time.length, v.length, a.length, d.length);

            const vData = v.slice(0, dataLength).filter((val): val is number => typeof val === 'number' && !isNaN(val));
            const aData = a.slice(0, dataLength).filter((val): val is number => typeof val === 'number' && !isNaN(val));
            const dData = d.slice(0, dataLength).filter((val): val is number => typeof val === 'number' && !isNaN(val));
            const timeData = time.slice(0, dataLength);

            const validIndices = time.map((_,i) => v[i] !== null && a[i] !== null && d[i] !== null).reduce((acc, cur, i) => cur ? [...acc, i] : acc, [] as number[]);
            const filteredV = validIndices.map(i => v[i] as number);
            const filteredA = validIndices.map(i => a[i] as number);
            const filteredD = validIndices.map(i => d[i] as number);
            const filteredTime = validIndices.map(i => time[i]);


            allV_plot.push(...filteredV);
            allA_plot.push(...filteredA);
            allD_plot.push(...filteredD);

            const hoverText = createHoverTextWithDescPlain(filteredTime, filteredV, filteredA, filteredD, word);
            const showTimeColorbar = !canShow4D && selectedWords.length <= 1;

            const colorbarOptions: Partial<ColorBar> | undefined = (canShow4D || showTimeColorbar) ? {
                title: canShow4D ? 'Sense Proportion' : 'Time (Year)',
                tickfont: COLORBAR_TICK_FONT,
                bgcolor: 'rgba(0,0,0,0)',
                outlinecolor: AXIS_LINE_COLOR,
                thickness: 15,
                len: 0.7,
            } : undefined;

            traces.push({
                x: filteredV,
                y: filteredA,
                z: filteredD,
                mode: 'lines+markers',
                type: 'scatter3d',
                name: word,
                text: hoverText,
                hoverinfo: 'text',
                line: {
                    color: traceColor,
                    width: selectedWords.length > 1 ? 2 : 1.5,
                    smoothing: 1.0
                },
                marker: {
                    size: selectedWords.length > 1 ? 3.5 : 4.5,
                    color: canShow4D ? senseProportions?.slice(0, filteredTime.length) : (showTimeColorbar ? filteredTime : traceColor),
                    colorscale: canShow4D ? SENSE_COLOR_SCALE : (showTimeColorbar ? TIME_COLOR_SCALE : undefined),
                    showscale: canShow4D || showTimeColorbar,
                    colorbar: colorbarOptions,
                    opacity: selectedWords.length > 1 ? 0.7 : 0.85
                }
            });
        });

        if (predictedVadSeries && selectedWords.length === 1 && traces.length > 0 && hasValidData) {
            const word = selectedWords[0];
            const actualTrace = traces.find(t => t.name === word);
            const wordData = allWordsData?.[word]?.temporal_vad;

            if (actualTrace && wordData && wordData.x.length > 0) {
                const lastActualTime = wordData.x[wordData.x.length -1];
                const lastActualV = wordData.v[wordData.v.length -1];
                const lastActualA = wordData.a[wordData.a.length -1];
                const lastActualD = wordData.d[wordData.d.length -1];

                if (lastActualV !== null && lastActualA !== null && lastActualD !== null) {
                    const forecastTimes = [lastActualTime, ...predictedVadSeries.map(p => p.time)];
                    const forecastV = [lastActualV, ...predictedVadSeries.map(p => p.v)];
                    const forecastA = [lastActualA, ...predictedVadSeries.map(p => p.a)];
                    const forecastD = [lastActualD, ...predictedVadSeries.map(p => p.d)];

                    allV_plot.push(...predictedVadSeries.map(p => p.v));
                    allA_plot.push(...predictedVadSeries.map(p => p.a));
                    allD_plot.push(...predictedVadSeries.map(p => p.d));

                    const forecastHoverText = createHoverTextWithDescPlain(forecastTimes, forecastV, forecastA, forecastD, word, true);

                    traces.push({
                        x: forecastV,
                        y: forecastA,
                        z: forecastD,
                        mode: 'lines+markers',
                        type: 'scatter3d',
                        name: `${word} (Forecast)`,
                        text: forecastHoverText,
                        hoverinfo: 'text',
                        line: {
                            color: actualTrace.line?.color || traceColors[0],
                            width: forecastLineStyle3D.width,
                            dash: forecastLineStyle3D.dash,
                            smoothing: 1.0,
                        },
                        marker: {
                            size: (actualTrace.marker?.size || 4.5) * 0.9,
                            color: actualTrace.line?.color || traceColors[0],
                            opacity: 0.9,
                            symbol: 'diamond'
                        },
                        showlegend: true
                    });
                }
            }
        }

        let dataRanges = { x: [0,1] as [number,number], y: [0,1] as [number,number], z: [0,1] as [number,number]};
        if (hasValidData || (predictedVadSeries && predictedVadSeries.length > 0)) {
            const getMinMaxWithFallback = (vals: number[], fallback: [number,number] = [0,1]): [number, number] => {
                const numericVals = vals.filter(v => typeof v === 'number' && !isNaN(v));
                if (numericVals.length === 0) return fallback;
                const min = Math.min(...numericVals);
                const max = Math.max(...numericVals);
                return [min, max];
            };
            dataRanges.x = getMinMaxWithFallback(allV_plot);
            dataRanges.y = getMinMaxWithFallback(allA_plot);
            dataRanges.z = getMinMaxWithFallback(allD_plot);

            const [xMin, xMax] = dataRanges.x;
            const [yMin, yMax] = dataRanges.y;
            const [zMin, zMax] = dataRanges.z;
            const xOffset = Math.max(0.05, (xMax - xMin) * 0.15);
            const yOffset = Math.max(0.05, (yMax - yMin) * 0.15);
            const zOffset = Math.max(0.05, (zMax - zMin) * 0.15);

            annotations.length = 0;
            annotations.push(
                { text: 'Pleasant', x: xMax + xOffset, y: (yMin + yMax) / 2, z: (zMin + zMax) / 2, showarrow: false, font: ANNOTATION_FONT, xanchor: 'left', yanchor: 'middle'},
                { text: 'Unpleasant', x: xMin - xOffset, y: (yMin + yMax) / 2, z: (zMin + zMax) / 2, showarrow: false, font: ANNOTATION_FONT, xanchor: 'right', yanchor: 'middle' },
                { text: 'Activated', x: (xMin + xMax) / 2, y: yMax + yOffset, z: (zMin + zMax) / 2, showarrow: false, font: ANNOTATION_FONT, xanchor: 'center', yanchor: 'bottom'},
                { text: 'Calm', x: (xMin + xMax) / 2, y: yMin - yOffset, z: (zMin + zMax) / 2, showarrow: false, font: ANNOTATION_FONT, xanchor: 'center', yanchor: 'top'},
                { text: 'In Control', x: (xMin + xMax) / 2, y: (yMin + yMax) / 2, z: zMax + zOffset, showarrow: false, font: ANNOTATION_FONT, xanchor: 'center', yanchor: 'middle', textangle: -90},
                { text: 'Controlled', x: (xMin + xMax) / 2, y: (yMin + yMax) / 2, z: zMin - zOffset, showarrow: false, font: ANNOTATION_FONT, xanchor: 'center', yanchor: 'middle', textangle: -90},
            );
        }
        return { traces, hasValidData: hasValidData || (predictedVadSeries && predictedVadSeries.length > 0 && selectedWords.length === 1), annotations, dataRanges };

    }, [selectedWords, allWordsData, is4D, senseProportions, predictedVadSeries]);


    if (selectedWords.length === 0) {
        return (
            <div className="plot-placeholder info">
                <h4>Select Word(s)</h4>
                <p>Select one or more words from the controls to view the 3D/4D plot.</p>
            </div>
        );
    }

    if (!plotMemoData.hasValidData) {
        return (
            <div className="plot-placeholder error">
                <Icon path={mdiAlertCircleOutline} size={1.6} className="placeholder-icon" />
                <h4>Data Error</h4>
                <p>No valid VAD data found for the selected word(s) to render the plot.</p>
            </div>
        );
    }

    const calculateRangeWithPadding = (minMax: [number, number], factor = 0.1): [number, number] => {
        const [min, max] = minMax;
        if (min === max) return [min - 0.1, max + 0.1];
        const range = max - min;
        const padding = Math.max(0.05, range * factor);
        return [min - padding, max + padding];
    };

    const xDisplayRange = calculateRangeWithPadding(plotMemoData.dataRanges.x);
    const yDisplayRange = calculateRangeWithPadding(plotMemoData.dataRanges.y);
    const zDisplayRange = calculateRangeWithPadding(plotMemoData.dataRanges.z);


    const commonAxisSettings = {
        backgroundcolor: AXIS_BG_COLOR,
        gridcolor: AXIS_GRID_COLOR,
        showbackground: true,
        zerolinecolor: AXIS_LINE_COLOR,
        tickfont: AXIS_TICK_FONT,
        linecolor: AXIS_LINE_COLOR,
        automargin: true,
        autorange: false,
    };

    const layout: Partial<Layout> = {
        autosize: true,
        height: undefined,
        width: undefined,
        margin: { l: 0, r: 0, b: 0, t: 0, pad: 4 },
        showlegend: selectedWords.length > 1 || (predictedVadSeries && selectedWords.length === 1),
        legend: {
            font: { size: 9 },
            yanchor: "top",
            y: 0.99,
            xanchor: "left",
            x: 0.01,
            bgcolor: 'rgba(255,255,255,0.7)',
            bordercolor: AXIS_LINE_COLOR,
            borderwidth: 1
        },
        scene: {
            xaxis: { ...commonAxisSettings, title: { text: 'Valence (V)', font: AXIS_TITLE_FONT }, range: xDisplayRange },
            yaxis: { ...commonAxisSettings, title: { text: 'Arousal (A)', font: AXIS_TITLE_FONT }, range: yDisplayRange },
            zaxis: { ...commonAxisSettings, title: { text: 'Dominance (D)', font: AXIS_TITLE_FONT }, range: zDisplayRange },
            camera: { eye: { x: 1.6, y: 1.6, z: 1.6 } },
            aspectmode: 'cube',
            annotations: plotMemoData.annotations
        },
        paper_bgcolor: PAPER_BG_COLOR,
        plot_bgcolor: PLOT_BG_COLOR,
        font: BASE_FONT,
        hoverlabel: {
            bgcolor: HOVER_BG_COLOR,
            bordercolor: HOVER_BORDER_COLOR,
            font: HOVER_FONT,
            align: 'left',
            namelength: -1
        }
    };

    return (
        <div className="plot3d-wrapper">
            <Plot
                data={plotMemoData.traces}
                layout={layout}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler={true}
                config={{ responsive: true, displaylogo: false, modeBarButtonsToRemove: ['toImage', 'sendDataToCloud', 'resetCameraDefault3d'] }}
            />
        </div>
    );
}

export default Plot3D4D;