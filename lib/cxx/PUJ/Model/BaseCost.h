// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__BaseCost__h__
#define __PUJ__BaseCost__h__

#include <cmath>
#include <vector>
#include <boost/program_options.hpp>

namespace PUJ
{
  namespace Model
  {
    /**
     */
    template< class _TModel >
    class BaseCost
    {
    public:
      using TModel  = _TModel;
      using TMatrix = typename _TModel::TMatrix;
      using TCol    = typename _TModel::TCol;
      using TRow    = typename _TModel::TRow;
      using TScalar = typename _TModel::TScalar;

    public:
      BaseCost( )
        : m_Model( nullptr )
        {
        }

      virtual void SetTrainData(
        const TMatrix& X, const TMatrix& Y,
        const PUJ::EInitValues& e = PUJ::Random
        )
        {
          if( this->m_BatchSize > 0 )
          {
            unsigned int n =
              ( unsigned int )(
                std::ceil( double( X.rows( ) ) / double( this->m_BatchSize ) )
                );
            for( unsigned int i = 0; i < n; ++i )
            {
              unsigned int bs = this->m_BatchSize;
              if( ( i + 1 ) * this->m_BatchSize > X.rows( ) )
                bs = X.rows( ) - ( i * this->m_BatchSize );
              this->m_X.push_back(
                X.block( i * this->m_BatchSize, 0, bs, X.cols( ) )
                );
              this->m_Y.push_back(
                Y.block( i * this->m_BatchSize, 0, bs, Y.cols( ) )
                );
            } // end for
          }
          else
          {
            this->m_X.push_back( X );
            this->m_Y.push_back( Y );
          } // end if
        }

      virtual ~BaseCost( ) = default;

      unsigned int GetBatchSize( ) const
        {
          return( this->m_BatchSize );
        }

      unsigned int GetNumberOfBatches( ) const
        {
          return( this->m_X.size( ) );
        }

      void SetBatchSize( unsigned int bs )
        {
          this->m_BatchSize = bs;
        }

      void SetModel( TModel* m )
        {
          this->m_Model = m;
        }

      virtual const TRow& GetParameters( ) const
        {
          return( this->m_Model->GetParameters( ) );
        }
      virtual TScalar operator()(
        unsigned int i, TRow* g = nullptr
        ) const = 0;
      virtual void operator-=( const TRow& g ) = 0;
      virtual void AddArguments(
        boost::program_options::options_description* o
        )
        {
          o->add_options( )
            (
              "batch_size",
              boost::program_options::value< unsigned int >( &this->m_BatchSize )->
              default_value( this->m_BatchSize ),
              "Batch size"
              );
        }

    protected:
      TModel* m_Model;
      unsigned int m_BatchSize { 0 };
      std::vector< TMatrix > m_X;
      std::vector< TMatrix > m_Y;
    };
  } // end namespace
} // end namespace

#endif // __PUJ__BaseCost__h__

// eof - $RCSfile$
