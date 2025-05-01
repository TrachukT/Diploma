import uuid
from sqlalchemy import Column, String, Float, TIMESTAMP, text
from sqlalchemy.dialects.postgresql import UUID
from .config import Base


class ModelMetrics(Base):
    __tablename__ = "model_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_type = Column(String, nullable=False)

    training_loss = Column(Float)
    training_accuracy = Column(Float)
    training_precision = Column(Float)
    training_recall = Column(Float)
    training_f1_score = Column(Float)

    evaluation_accuracy = Column(Float)
    evaluation_precision = Column(Float)
    evaluation_recall = Column(Float)
    evaluation_f1_score = Column(Float)

    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
